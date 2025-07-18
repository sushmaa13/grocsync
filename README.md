# 🛒 GrocSync — Intelligent Grocery Management System
GrocSync introduces an intelligent pipeline for streamlined grocery management using OCR, machine learning, and data integration techniques. The system is built to reduce waste, track inventory, predict future needs, and recommend recipes using available ingredients.


 🚀 Key Features

- 📷 **Receipt Upload and OCR Processing** 
  Users can upload scanned grocery bills. Tesseract OCR extracts item names, quantities, prices, and expiry dates from the images.

- 🧾 **Data Integration & Inventory Management**  
  Inventory is maintained in structured CSV files. If an item exists, quantity and cost are updated. New items are added via pandas-based merging logic.

- ⏳ **Expiry Date Tracking**  
  Items are tagged with expiry dates. A monitoring module sends alerts for items nearing expiration, promoting timely consumption or disposal.

- ✍️ **User Feedback Integration**  
  Users provide feedback on item usage. This data is logged and used to retrain the ML model for more accurate future predictions.

- 🧠 **ML-Based Prediction Engine**  
  to predict items likely needed for the next cycle, based on usage patterns.

- 🍽️ **Recipe Recommendation System**  
  Using a curated recipe dataset, recipes matching current inventory are recommended to reduce waste and encourage home cooking.

- 📊 **Export & Visualization**  
  Inventory and prediction results are exported to Excel. Data visualizations can be generated using `matplotlib` or `seaborn`.





