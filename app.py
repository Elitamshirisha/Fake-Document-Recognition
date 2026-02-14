import streamlit as st
import sqlite3  # Import SQLite module
import os
import base64
import subprocess

# Function to convert a file to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-position: center;
        background-size: cover;
        font-family: "Times New Roman", serif;
    }}
    h1, h2, h3, p {{
        font-family: "Times New Roman", serif;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Set the background image for the app
set_background('Background/1.png')


# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect('users.db')  # Connect to the SQLite database
    cursor = conn.cursor()
    
    # Create the users table if it does not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY, 
                        password TEXT)''')
    
    conn.commit()
    conn.close()


# Function to render the home page
def home():
    """
    Renders the Home page.
    Greets and provides information about the app.
    """
    st.markdown("<h1 style='text-align: center;'>Fake Document Recognition </h1>", unsafe_allow_html=True)

    # Animation
    animation_path = "background/1.gif"  # Relative path to your GIF
    if os.path.exists(animation_path):  # Ensure the file exists
        animation_base64 = get_base64(animation_path)
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
          <img src="data:image/gif;base64,{animation_base64}" alt="Fitness Animation" style="max-width: 400px;"/>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Error: Animation file not found at '{animation_path}'")

    st.markdown("<p style='text-align: center;'>Welcome to our Fake Document Recognition ! </p>", unsafe_allow_html=True)


# Function to register a user
def signup():
    """
    Renders the sign-up page for the web application.
    Allows a new user to create an account by providing a username and password.
    Checks for existing username and password confirmation before adding the user.
    """
    st.markdown("<h2 style='text-align: center;'>Create a New Account</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please enter your details to create a new account:</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Sign Up"):
            # Initialize the database if not done already
            init_db()
            
            # Check if username already exists
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()

            if user:
                st.error("Username already exists. Please choose a different username.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                # Insert new user into the database
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                conn.commit()
                st.success("You have successfully signed up!")
                st.info("Please go to the Login page to log in.")
            
            conn.close()


# Function to log in a user
def login():
    """
    Renders the login page for the web application.
    Allows an existing user to log in by providing their username and password.
    Checks for the existence of the username and matches the password.
    """
    st.markdown("<h2 style='text-align: center;'>Login Section</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please enter your credentials to log in:</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # Initialize the database if not done already
            init_db()
            
            # Verify the credentials
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
            user = cursor.fetchone()

            if user:
                st.success(f"Welcome {username.title()}!")
                st.write("You have successfully logged in.")
                # Assuming app1.py is the app you want to launch after login
                subprocess.run(["streamlit", "run", "app1.py"])
            else:
                st.error("Incorrect username or password. Please try again.")
            
            conn.close()


# Main function to run the Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    Provides navigation between the sign-up and login pages using a sidebar dropdown menu list.
    """
    # Sidebar navigation
    choice = st.sidebar.radio("Menu", ["Home", "Login", "Sign Up"], index=0)  # Default to 'Home'

    if choice == "Sign Up":
        signup()
    elif choice == "Login":
        login()
    elif choice == "Home":
        home()


if __name__ == "__main__":
    main()
