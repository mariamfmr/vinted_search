from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import sqlite3
import hashlib
from werkzeug.utils import secure_filename
from modules.image_search_utils import pattern_detector, image_search_utils
from vinted_scraper import VintedScraper

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE'] = 'database/login.db'


vinted = VintedScraper(baseurl='https://www.vinted.com')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def connect_db():
    return sqlite3.connect('database/login.db')
    

def init_db():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS saved_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        item_name TEXT, -- Add this column if needed
        item_url TEXT NOT NULL,
        image_url TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    ''')

    conn.commit()
    conn.close()

init_db()
if __name__ == "__main__":
    init_db()


@app.route('/')
def start():
    return render_template('01_start.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        try:
            with connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                user = cursor.fetchone()

            if user and user[2] == password:
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('category_main'))
            else:
                flash('Invalid username or password. Please try again.', 'error')
                return redirect(url_for('login'))
        except sqlite3.Error as e:
            flash(f'A database error occurred: {str(e)}', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Enhanced validations
        if not username or not password:
            flash('Username and password cannot be empty.', 'error')
            return redirect(url_for('register'))

        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = hash_password(password)

        try:
            with connect_db() as conn:
                cursor = conn.cursor()
                
                # Check if username already exists
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                existing_user = cursor.fetchone()

                if existing_user:
                    flash('Username already exists. Please choose another one.', 'error')
                    return redirect(url_for('register'))

                # Insert new user into the database
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()

            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))

        except sqlite3.Error as e:
            flash(f'A database error occurred: {str(e)}', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('start'))

# Update the save_item route in app.py
# Update the save_item route in app.py
# Update the save_item route in app.py
@app.route('/save-item', methods=['POST'])
def save_item():
    if 'username' not in session:
        return {'status': 'error', 'message': 'Please log in to save items.'}, 401

    try:
        data = request.get_json()
        item_url = data.get('url')
        image_url = data.get('image_url')
        item_name = data.get('title')  # Get the title from the request

        if not item_url or not image_url or not item_name:
            return {'status': 'error', 'message': 'Missing required item information'}, 400

        username = session['username']
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user:
                user_id = user[0]
                # Check if item already exists for this user
                cursor.execute("SELECT id FROM saved_items WHERE user_id = ? AND item_url = ?", 
                             (user_id, item_url))
                existing_item = cursor.fetchone()
                
                if existing_item:
                    return {'status': 'error', 'message': 'Item already saved'}, 400

                # Save the item with all details including the title
                cursor.execute("""
                    INSERT INTO saved_items (user_id, item_name, item_url, image_url) 
                    VALUES (?, ?, ?, ?)""", 
                    (user_id, item_name, item_url, image_url))
                conn.commit()
                return {'status': 'success', 'message': 'Item saved successfully'}, 200
            else:
                return {'status': 'error', 'message': 'User not found'}, 404

    except Exception as e:
        print(f"Error saving item: {str(e)}")  # For debugging
        return {'status': 'error', 'message': 'Failed to save item'}, 500
    
@app.route('/saved_items')
def saved_items():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            # Get user_id
            cursor.execute("SELECT id FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()
            
            if user:
                user_id = user[0]
                # Get all saved items for this user
                cursor.execute("""
                    SELECT id, item_name, item_url, image_url 
                    FROM saved_items 
                    WHERE user_id = ?
                    ORDER BY id DESC""", (user_id,))
                
                saved_items = []
                for row in cursor.fetchall():
                    saved_items.append({
                        'id': row[0],
                        'title': row[1],
                        'url': row[2],
                        'image_url': row[3]
                    })
                
                return render_template('saved_items.html', 
                                     saved_items=saved_items, 
                                     username=session['username'])
            else:
                flash("User not found", "error")
                return redirect(url_for('login'))
    except sqlite3.Error as e:
        flash(f"Database error: {str(e)}", "error")
        return redirect(url_for('login'))
    
@app.route('/delete-item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    if 'username' not in session:
        return {'status': 'error', 'message': 'Please log in to delete items.'}, 401

    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (session['username'],))
        user = cursor.fetchone()

        if user:
            user_id = user[0]
            cursor.execute("DELETE FROM saved_items WHERE id = ? AND user_id = ?", 
                         (item_id, user_id))
            conn.commit()
            return {'status': 'success', 'message': 'Item deleted successfully'}, 200
        else:
            return {'status': 'error', 'message': 'User not found'}, 404
        
@app.route('/category')
def category_main():
    if 'username' not in session:
        return redirect(url_for('login'))
    main_categories = ["Women's Clothing", "Men's Clothing", "Children's Clothing"]
    return render_template('02_category_main.html', main_categories=main_categories, username=session['username'])

@app.route('/category/<main_category>')
def category_subcategory(main_category):
    """Display subcategories for the selected main category."""
    subcategories = {
        "Women's Clothing": ["Tops", "Dresses", "Outerwear", "Activewear", "Sleepwear", "Swimwear", "Jackets", "Sweaters", "Blouses", "Skirts", "Pants", "Jeans", "Shorts", "Suits", "Jumpsuits", "Leggings", "Accessories", "Shoes", "Bags", "Other"],
        "Men's Clothing": ["Tops", "Bottoms", "Suits", "Outerwear", "Activewear", "Sleepwear", "Swimwear", "Jackets", "Sweaters", "Shirts", "Pants", "Jeans", "Shorts", "Accessories", "Shoes", "Bags", "Other"],
        "Children's Clothing": ["Tops", "Bottoms", "Dresses", "Outerwear", "Activewear", "Sleepwear", "Swimwear"]
    }
    return render_template('03_category_subcategory.html', main_category=main_category, subcategories=subcategories.get(main_category, []), username=session['username'])

@app.route('/upload/<main_category>/<subcategory>', methods=['GET', 'POST'])
def upload(main_category, subcategory):
    """
    Handle image uploads for a specific subcategory.
    """
    if request.method == 'POST':
        print(f"Debug: Received POST request for {main_category} > {subcategory}")
        
        # Check for the image in the request
        if 'image' not in request.files:
            print("Debug: No file part in the request")
            return "No file part", 400

        file = request.files['image']

        if file.filename == '':
            print("Debug: No file selected")
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Debug: File saved to {filepath}")

            # Get sizes from the form
            selected_sizes = request.form.getlist('size')
            custom_size = request.form.get('custom_size')

            # Combine sizes into a single list
            sizes = selected_sizes if selected_sizes else []
            if custom_size and custom_size.strip():
                sizes.append(custom_size)
            
            print(f"Debug: Selected sizes: {sizes}")

            # Redirect to results with query parameters
            return redirect(url_for('results', query=filename, main_category=main_category, subcategory=subcategory, sizes=",".join(sizes), username=session['username']))

    return render_template('04_upload.html', main_category=main_category, subcategory=subcategory, username=session['username'])

@app.route('/results')
def results():
    """
    Handle the results page, calculating similarity and displaying the most relevant items.
    """
    query = request.args.get('query')
    main_category = request.args.get('main_category')
    subcategory = request.args.get('subcategory')
    sizes = request.args.get('sizes', '').split(',') if request.args.get('sizes') else []
    print(f"Debug: Received query {query} for {main_category} > {subcategory} with sizes {sizes}")

    if not query:
        return "No query provided", 400

    # Construct the full path to the uploaded image
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], query)
    if not os.path.exists(uploaded_image_path):
        return "Uploaded file not found", 404

    try:
        # Extract features from the uploaded image
        uploaded_color_weights = image_search_utils.get_color_weights(uploaded_image_path)
        uploaded_pattern_weights = pattern_detector.detect_pattern(uploaded_image_path)

        if not uploaded_color_weights:
            return "Failed to extract features from the uploaded image", 500

        # Fetch color with the highest weight
        color = max(uploaded_color_weights, key=uploaded_color_weights.get)

        # For each size, add a search query with the size
        search_results = []
        
        for size in sizes:
            search_query = f"{color} {size} {main_category} {subcategory}"
            search_results += vinted.search({'search_text': search_query})
            


        # Filter results by size if sizes were selected
        if sizes and sizes[0]:
            search_results = [
                item for item in search_results 
                if any(size.lower() in item.title.lower() for size in sizes)
            ]
            

        print(f"Search parameters: {search_query}")
        print(f"Debug: Found {len(search_results)} items for comparison")

        items_with_scores = []

        for item in search_results:
            try:
                item_color_weights = image_search_utils.get_color_weights(item.photos[0].url)
                item_pattern_weights = pattern_detector.detect_pattern(item.photos[0].url)

                if not item_color_weights:
                    continue

                similarity = image_search_utils.calculate_combined_similarity(
                    uploaded_color_weights, item_color_weights,
                    uploaded_pattern_weights, item_pattern_weights
                )

                print(f"Debug: Calculated similarity for {item.title}: {similarity}")

                items_with_scores.append({
                    'title': item.title,
                    'url': item.url,
                    'image_url': item.photos[0].url,
                    'similarity': similarity
                })

            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        sorted_items = sorted(items_with_scores, key=lambda x: x['similarity'], reverse=True)

        return render_template(
            '05_results.html',
            query=query,
            uploaded_image=uploaded_image_path, username = session['username'],
            results=sorted_items[:10],
        )

    except Exception as e:
        print(f"Error processing results: {e}")
        return "An error occurred while processing the results", 500

# Rest of the code remains the same
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)