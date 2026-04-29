
# Project Execution Steps

## Bearing_Fault_Diagnosis

Follow the steps carefully to run the project successfully.

---

## Step 1: Download the Repository

* Download the project ZIP file
  **OR**
* Clone using git:

```bash
git clone https://github.com/vijayvardhan-killi/LiteFDNet.git
```

---

## Step 2: Extract the Project (If ZIP)

* Right click the ZIP file
* Click **Extract Here** or **Extract All**
* Open the extracted project folder

---

## step 3: Create a Virtual Environment

Open terminal inside the project folder and run:

### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

Wait until all packages are installed successfully.

---

## Step 5: Setup Database Using XAMPP

1. Open **XAMPP Control Panel**
2. Start:

   * Apache
   * MySQL
3. Click **Admin** next to MySQL
   (This opens phpMyAdmin)

### Create Database:

* Click **New**
* Enter database name:

```
Bearing_Fault_Diagnosis
```

* Click **Create**

---

## Step 6: Run Migrations

Make sure `settings.py` is already configured with MySQL.

Then run:

```bash
python manage.py makemigrations
python manage.py migrate
```

This will create all tables inside `Bearing_Fault_Diagnosis` database.

---

## Step 7: Start the Server
  python manage.py runserver <port>

```bash
python manage.py runserver 1234
```

Open browser and go to:

```
http://127.0.0.1:8000/
```

---

If everything runs without errors, the project is successfully set up.

---



---

# Running the Project (After Initial Setup)

Once the project has been set up for the first time, follow these steps to run it again.

---

## Step 1: Start XAMPP

1. Open **XAMPP Control Panel**
2. Start:

   *  Apache
   *  MySQL

Make sure both are running.

---

## Step 2: Activate Virtual Environment

Open terminal inside the project folder.

### Windows:

```bash
venv\Scripts\activate
```

### Mac/Linux:

```bash
source venv/bin/activate
```

---

## Step 3: Apply Migrations (If Needed)

```bash
python manage.py makemigrations
python manage.py migrate
```

*(If no model changes were made, Django will show “No changes detected” — that’s normal.)*

---

## Step 4: Run the Server

```bash
python manage.py runserver 1234
```

Open browser:

```
http://127.0.0.1:1234/
```

---
The project should now be running successfully.
