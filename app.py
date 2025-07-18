from flask import Flask, request, redirect,render_template, url_for, session,send_file, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from utils.ocr_utils import extract_text, parse_items
from ml import grocery_predictor
import pickle



app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

BILLS_FOLDER = os.path.join('data', 'bills')
os.makedirs(BILLS_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('bills')
        all_items = session.get('grocery_items', [])
        for file in files:
            if file and file.filename != "":
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filepath)
                text = extract_text(filepath)
                items = parse_items(text)
                all_items.extend(items)
        session['grocery_items'] = all_items
        return redirect('/confirm_merge')
    return render_template('index.html')

@app.route('/review_items', methods=['GET', 'POST'])
def review_items():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("images")
        all_items = []
        for file in uploaded_files:
            if file.filename == "":
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text = extract_text(filepath)
            items = parse_items(text)
            all_items.extend(items)
        session['grocery_items'] = all_items

    items = session.get('grocery_items', [])
    return render_template('review_items.html', items=items)

@app.route('/add_manual', methods=['POST'])
def add_manual():
    new_item = {
        'name': request.form['name'],
        'quantity': int(request.form['quantity']),
        'cost': float(request.form['cost']),
        'expiry': request.form['expiry']
    }
    grocery_items = session.get('grocery_items', [])
    grocery_items.append(new_item)
    session['grocery_items'] = grocery_items
    return redirect('/review_items')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    merged_file = os.path.join('data', 'bills', 'merged_bills.csv')
    final_file = os.path.join('data', 'bills', 'final_data.csv')

    if request.method == 'POST':
        form_data = request.form
        feedback_entries = []

        index = 1
        while f'name_{index}' in form_data:
            item_name = form_data.get(f'name_{index}')
            quantity_bought = form_data.get(f'quantity_{index}')
            status = form_data.get(f'status_{index}')
            used_quantity = form_data.get(f'used_quantity_{index}', '')

            if status == 'Used':
                used_quantity = quantity_bought
            elif status == 'Expired':
                used_quantity = 0

            feedback_entries.append({
                'Item Name': item_name,
                'Quantity Bought': quantity_bought,
                'Status': status,
                'Used Quantity': used_quantity
            })

            index += 1

        if os.path.exists(final_file):
            existing = pd.read_csv(final_file)
            updated = pd.concat([existing, pd.DataFrame(feedback_entries)], ignore_index=True)
        else:
            updated = pd.DataFrame(feedback_entries)

        updated.to_csv(final_file, index=False)
        return redirect(url_for('submit_final_data'))  # ✅ Redirect to trigger prediction

    if os.path.exists(merged_file):
        merged_df = pd.read_csv(merged_file, index_col=False)
        merged_df = merged_df.rename(columns={
            'name': 'Item Name',
            'quantity': 'Quantity Bought',
            'cost': 'Cost',
            'expiry': 'Expiry'
        })
    else:
        merged_df = pd.DataFrame(columns=['Item Name', 'Quantity Bought', 'Cost', 'Expiry'])

    if os.path.exists(final_file):
        feedback_df = pd.read_csv(final_file)
        submitted_names = feedback_df['Item Name'].tolist()
    else:
        submitted_names = []

    unsubmitted = merged_df[~merged_df['Item Name'].isin(submitted_names)]
    items = unsubmitted.to_dict(orient='records')
    return render_template('feedback.html', items=items)

# ✅ New route to trigger prediction
@app.route('/submit_final_data')
def submit_final_data():
    try:
        grocery_predictor.main()
        return redirect(url_for('predict_next_bill'))  # Redirect to results
    except Exception as e:
        return f"Prediction failed: {e}", 500

@app.route('/success')
def success_page():
    return render_template('success.html')

@app.route('/stored_bills')
def stored_bills():
    merged_bills_file_path = os.path.join('data', 'bills', 'merged_bills.csv')
    items = []

    if os.path.exists(merged_bills_file_path):
        with open(merged_bills_file_path, newline='') as file:
            reader = csv.DictReader(file)
            items = list(reader)

    return render_template('stored_bills.html', items=items, enumerate=enumerate)

@app.route('/delete_stored_item', methods=['POST'])
def delete_stored_item():
    index = int(request.form.get('index'))
    file_path = os.path.join('data', 'bills', 'merged_bills.csv')

    with open(file_path, 'r') as file:
        reader = list(csv.reader(file))
        header = reader[0]
        rows = reader[1:]

    if 0 <= index < len(rows):
        del rows[index]

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

    return redirect('/stored_bills')

@app.route('/delete_bill/<filename>', methods=['GET', 'POST'])
def delete_bill(filename):
    bill_path = os.path.join('data', 'bills', filename)
    if os.path.exists(bill_path):
        os.remove(bill_path)
    return redirect(url_for('stored_bills'))

@app.route('/confirm_merge', methods=['GET', 'POST'])
def confirm_merge():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'merge':
            items = session.get('grocery_items', [])

            expiry_dates = request.form.getlist('expiry[]')
            for i in range(len(items)):
                items[i]['expiry'] = expiry_dates[i]
                items[i]['status'] = 'fresh'

            manual_names = request.form.getlist('manual_name[]')
            manual_quantities = request.form.getlist('manual_quantity[]')
            manual_costs = request.form.getlist('manual_cost[]')
            manual_expiries = request.form.getlist('manual_expiry[]')

            for i in range(len(manual_names)):
                manual_item = {
                    'name': manual_names[i],
                    'quantity': manual_quantities[i],
                    'cost': manual_costs[i],
                    'expiry': manual_expiries[i],
                    'status': 'fresh'
                }
                items.append(manual_item)

            merged_bills_file_path = os.path.join('data', 'bills', 'merged_bills.csv')
            with open(merged_bills_file_path, 'a', newline='') as file:
                fieldnames = ['name', 'quantity', 'cost', 'expiry', 'status']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerows(items)

            session.pop('grocery_items', None)
            return redirect('/stored_bills')

        elif action == 'discard':
            session.pop('grocery_items', None)
            items = []
            return render_template('confirm_merge.html', items=items)

    items = session.get('grocery_items', [])
    return render_template('confirm_merge.html', items=items)

@app.route('/expiry_status/<filename>')
def expiry_status(filename):
    bill_items = []
    bills_file_path = os.path.join('data', 'bills', 'final_groceries.csv')

    try:
        with open(bills_file_path, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['name'] == filename:
                    bill_items.append(row)

    except FileNotFoundError:
        return "No bills found.", 404

    return render_template('expiry_status.html', bill_items=bill_items, filename=filename)

@app.route('/view_expiry_status')
def view_expiry_status():
    file_path = os.path.join('data', 'bills', 'merged_bills.csv')

    if not os.path.exists(file_path):
        return "No merged bills found yet. Please upload or merge a bill first."

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        items = [row for row in reader]

    current_date = datetime.today()
    expired_items = []
    nearing_expiry_items = []
    fresh_items = []

    for item in items:
        try:
            expiry_date = datetime.strptime(item['expiry'], '%Y-%m-%d')
        except ValueError:
            continue

        if expiry_date < current_date:
            expired_items.append(item)
        elif expiry_date <= current_date + timedelta(days=7):
            nearing_expiry_items.append(item)
        else:
            fresh_items.append(item)

    return render_template('view_expiry_status.html',
                           expired_items=expired_items,
                           nearing_expiry_items=nearing_expiry_items,
                           fresh_items=fresh_items)

@app.route('/predict-next-bill')
def predict_next_bill():
    # Run prediction before loading the result CSV
    grocery_predictor.main()  # ✅ This runs your prediction script

    bills_path = r"C:\Users\Tejasriseelam\Desktop\grocery_app\data\bills"
    csv_path = os.path.join(bills_path, "predicted_bill.csv")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error loading prediction results: {e}"

    records = df.to_dict(orient='records')
    columns = df.columns.values

    return render_template('predicted_bill.html', records=records, columns=columns)
@app.route('/download-bill')
def download_bill():
    import pandas as pd

    bills_path = r"C:\Users\Tejasriseelam\Desktop\grocery_app\data\bills"
    predicted_path = os.path.join(bills_path, "predicted_bill.csv")
    temp_path = os.path.join(bills_path, "download_bill.csv")

    try:
        df = pd.read_csv(predicted_path)

        # Remove summary row if present
        df = df[df['Item Name'].str.lower() != 'total savings']

        # Keep only needed columns
        df = df[['Item Name', 'Required Quantity']]

        # Remove rows where required quantity is 0 or less
        df = df[df['Required Quantity'] > 0]

        # Optional: round quantity to whole number if needed
        df['Required Quantity'] = df['Required Quantity'].round().astype(int)

        # Save to new CSV for download
        df.to_csv(temp_path, index=False)

    except Exception as e:
        return f"Error generating download: {e}"

    return send_file(temp_path, as_attachment=True)

# Load model and vectorizer
with open("recipe_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/generate", methods=["GET", "POST"])
def generate():
    recipe = None
    if request.method == "POST":
        ingredients = request.form["ingredients"]
        input_vec = vectorizer.transform([ingredients])
        recipe = model.predict(input_vec)[0]
    return render_template("generate.html", recipe=recipe)


@app.route('/visuals')
def show_visuals():
    # Load combined data
    data = pd.read_csv(r'data\bills\combined_data.csv')

    # Ensure numeric columns are proper type
    for col in ['quantity', 'cost', 'used_quantity', 'quantity_bought']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Parse expiry dates
    data['expiry'] = pd.to_datetime(data['expiry'], errors='coerce')

    # Add month column for grouping by month/year
    data['month'] = data['expiry'].dt.to_period('M').astype(str)

    # Calculate total used and wasted quantities
    total_used = data['used_quantity'].sum()
    total_bought = data['quantity_bought'].sum()
    total_wasted = max(total_bought - total_used, 0)

    # Pie Chart: Used vs Wasted
    if total_used + total_wasted > 0:
        plt.figure(figsize=(5, 5))
        plt.pie([total_used, total_wasted], labels=['Used', 'Wasted'], autopct='%1.1f%%',
                colors=['#4CAF50', '#E57373'], startangle=90)
        plt.title('Used vs Wasted Groceries')
        plt.savefig('static/used_vs_wasted.png')
        plt.close()
    else:
        plt.figure(figsize=(5,5))
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig('static/used_vs_wasted.png')
        plt.close()

    # Monthly Spending Trend
    monthly_spending = data.groupby('month')['cost'].sum().reset_index()
    if not monthly_spending.empty:
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=monthly_spending, x='month', y='cost', marker='o', color='#1976D2')
        plt.title('Monthly Spending Trend')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/spending_trend.png')
        plt.close()
    else:
        plt.figure(figsize=(8,4))
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig('static/spending_trend.png')
        plt.close()

    # Estimate money wasted
    data['money_wasted'] = 0
    valid = data['quantity_bought'] > 0
    data.loc[valid, 'money_wasted'] = (data.loc[valid, 'cost'] * 
                                      (data.loc[valid, 'quantity_bought'] - data.loc[valid, 'used_quantity']) / 
                                      data.loc[valid, 'quantity_bought'])

    # Monthly Money Wasted
    monthly_wasted = data.groupby('month')['money_wasted'].sum().reset_index()
    if not monthly_wasted.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=monthly_wasted, x='month', y='money_wasted', color='#FFA726')
        plt.title('Estimated Money Wasted Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/money_wasted.png')
        plt.close()
    else:
        plt.figure(figsize=(8,4))
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig('static/money_wasted.png')
        plt.close()

    # Top 5 wasted items
    wasted_items = data.groupby('name')['money_wasted'].sum().sort_values(ascending=False).head(5).reset_index()
    if not wasted_items.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=wasted_items, x='money_wasted', y='name', palette='Reds_r')
        plt.title('Top 5 Items by Estimated Money Wasted')
        plt.xlabel('Money Wasted')
        plt.ylabel('Item Name')
        plt.tight_layout()
        plt.savefig('static/top_wasted_items.png')
        plt.close()
    else:
        plt.figure(figsize=(8,4))
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig('static/top_wasted_items.png')
        plt.close()

    # Read total savings from savings.csv
    try:
        savings_df = pd.read_csv(r'data\bills\savings.csv')
        total_savings = float(savings_df['Total Savings'].iloc[0])
    except Exception:
        total_savings = 0

    # Create a bar chart for Total Savings (since no monthly data)
    plt.figure(figsize=(6,4))
    plt.bar(['Total Savings'], [total_savings], color='#66BB6A')
    plt.title('Total Savings')
    plt.ylabel('Amount')
    plt.tight_layout()
    plt.savefig('static/money_saved.png')
    plt.close()

    return render_template('visuals.html', total_savings=total_savings)

if __name__ == "__main__":
    app.run(debug=True)
