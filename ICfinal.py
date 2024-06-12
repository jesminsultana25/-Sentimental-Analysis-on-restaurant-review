import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np

# Sample DataFrame with sentiment analysis results for multiple restaurants
df = pd.read_csv('cleaned_and_analyzed_dataset.csv')  # Assuming you've saved the cleaned and analyzed dataset

# Create a function to assign aspect labels based on review content
def assign_aspect_label(review):
    if isinstance(review, str):  # Check if the review is a string
        if "ambience" in review.lower() or "ambiance" in review.lower():
            return "Ambience"
        elif "service" in review.lower() or "waiter" in review.lower() or "staff" in review.lower():
            return "Service"
        elif "food" in review.lower() or "taste" in review.lower() or "starrers" in review.lower():
            return "Food Taste"
        elif "price" in review.lower() or "value" in review.lower():
            return "Price/Value"
        elif "hygiene" in review.lower() or "cleanliness" in review.lower() or "hospitalized" in review.lower() or "infection" in review.lower():
            return "Hygiene/Cleanliness"
        elif "experience" in review.lower() or "overall" in review.lower() or "beyond" in review.lower():
            return "Overall Experience"
        else:
            return "Other"
    else:
        return np.nan  # Return NaN for non-string values

# Assign aspect labels to each review
df['aspect_label'] = df['Review'].apply(assign_aspect_label)

# Function to calculate sentiment ratio
def calculate_sentiment_ratio(restaurant_df):
    sentiment_counts = restaurant_df['predicted_sentiment'].value_counts()
    if 'positive' in sentiment_counts:
        positive_count = sentiment_counts['positive']
    else:
        positive_count = 0
    if 'negative' in sentiment_counts:
        negative_count = sentiment_counts['negative']
    else:
        negative_count = 0
    total_count = positive_count + negative_count
    if total_count == 0:
        return 0
    return positive_count / total_count

# Function to perform competitive analysis
def perform_competitive_analysis():
    selected_restaurant = restaurant_var.get()
    selected_competitor = competitor_var.get()
    if selected_restaurant == "Select Restaurant":
        tk.messagebox.showinfo("Error", "Please select a restaurant.")
    elif selected_competitor == "Select Competitor":
        tk.messagebox.showinfo("Error", "Please select a competitor.")
    else:
        results_window = tk.Toplevel(root)
        results_window.title(f"Competitive Analysis: {selected_restaurant} vs {selected_competitor}")

        # Filter DataFrame for the selected restaurant and competitor
        restaurant_df = df[df['Restaurant'] == selected_restaurant]
        competitor_df = df[df['Restaurant'] == selected_competitor]

        # Calculate sentiment ratio for the selected restaurant and competitor
        restaurant_sentiment_ratio = calculate_sentiment_ratio(restaurant_df)
        competitor_sentiment_ratio = calculate_sentiment_ratio(competitor_df)

        # Display sentiment ratio comparison
        comparison_text = f"Sentiment Ratio Comparison:\n{selected_restaurant}: {restaurant_sentiment_ratio:.2f}\n{selected_competitor}: {competitor_sentiment_ratio:.2f}"
        comparison_label = tk.Label(results_window, text=comparison_text)
        comparison_label.pack()

        # Analyze sentiment based on different aspects
        aspect_sentiment_analysis_label = tk.Label(results_window, text="Sentiment Analysis by Aspect:")
        aspect_sentiment_analysis_label.pack()

        # Calculate sentiment ratios for different aspects
        aspects = df['aspect_label'].unique()
        for aspect in aspects:
            restaurant_aspect_df = restaurant_df[restaurant_df['aspect_label'] == aspect]
            competitor_aspect_df = competitor_df[competitor_df['aspect_label'] == aspect]
            restaurant_aspect_sentiment_ratio = calculate_sentiment_ratio(restaurant_aspect_df)
            competitor_aspect_sentiment_ratio = calculate_sentiment_ratio(competitor_aspect_df)
            aspect_comparison_text = f"{aspect}:\n{selected_restaurant}: {restaurant_aspect_sentiment_ratio:.2f}\n{selected_competitor}: {competitor_aspect_sentiment_ratio:.2f}"
            aspect_comparison_label = tk.Label(results_window, text=aspect_comparison_text)
            aspect_comparison_label.pack()

        # Add explanations for why one restaurant has more positive comments than the other
        positive_comments_explanation_label = tk.Label(results_window, text="Explanations for More Positive Comments:")
        positive_comments_explanation_label.pack()

        if restaurant_sentiment_ratio > competitor_sentiment_ratio:
            explanation_text = f"{selected_restaurant} has more positive comments due to better {aspects[0]}, {aspects[1]}, and {aspects[2]}."
        elif restaurant_sentiment_ratio < competitor_sentiment_ratio:
            explanation_text = f"{selected_competitor} has more positive comments due to better {aspects[0]}, {aspects[1]}, and {aspects[2]}."
        else:
            explanation_text = "Both restaurants have similar overall sentiment ratios."
        explanation_label = tk.Label(results_window, text=explanation_text)
        explanation_label.pack()

        # Add explanations for why one restaurant has fewer positive comments than the other
        fewer_positive_comments_explanation_label = tk.Label(results_window, text="Explanations for Fewer Positive Comments:")
        fewer_positive_comments_explanation_label.pack()

        if restaurant_sentiment_ratio < competitor_sentiment_ratio:
            explanation_text = f"{selected_restaurant} has fewer positive comments due to inferior {aspects[0]}, {aspects[1]}, and {aspects[2]}."
        elif restaurant_sentiment_ratio > competitor_sentiment_ratio:
            explanation_text = f"{selected_competitor} has fewer positive comments due to inferior {aspects[0]}, {aspects[1]}, and {aspects[2]}."
        else:
            explanation_text = "Both restaurants have similar overall sentiment ratios."
        explanation_label = tk.Label(results_window, text=explanation_text)
        explanation_label.pack()

        # Function to return to the main page
        def return_to_main_page(window):
            window.destroy()
            root.deiconify()
        # Add a button to return to the main page
        return_button = tk.Button(results_window, text="Return to Main Page", command=lambda: return_to_main_page(results_window))
        return_button.pack()
# Function to perform individual analysis
def perform_individual_analysis():
    selected_restaurant = restaurant_var.get()
    if selected_restaurant == "Select Restaurant":
        tk.messagebox.showinfo("Error", "Please select a restaurant.")
    else:
        individual_analysis_window = tk.Toplevel(root)
        individual_analysis_window.title(f"Individual Analysis: {selected_restaurant}")

        # Filter DataFrame for the selected restaurant
        restaurant_df = df[df['Restaurant'] == selected_restaurant]

        # Calculate overall sentiment analysis
        overall_sentiment_ratio = calculate_sentiment_ratio(restaurant_df)
        overall_sentiment_label = tk.Label(individual_analysis_window, text=f"Overall Sentiment Ratio: {overall_sentiment_ratio:.2f}")
        overall_sentiment_label.pack()

        # Get top 5 positive reviews
        top_positive_reviews = restaurant_df[restaurant_df['predicted_sentiment'] == 'positive'].head(5)
        positive_reviews_label = tk.Label(individual_analysis_window, text="Top 5 Positive Reviews:")
        positive_reviews_label.pack()
        for idx, review in top_positive_reviews.iterrows():
            review_text = review['Review']
            review_label = tk.Label(individual_analysis_window, text=f"{idx+1}. {review_text}")
            review_label.pack()

        # Get top 5 negative reviews
        top_negative_reviews = restaurant_df[restaurant_df['predicted_sentiment'] == 'negative'].head(5)
        negative_reviews_label = tk.Label(individual_analysis_window, text="Top 5 Negative Reviews:")
        negative_reviews_label.pack()
        for idx, review in top_negative_reviews.iterrows():
            review_text = review['Review']
            review_label = tk.Label(individual_analysis_window, text=f"{idx+1}. {review_text}")
            review_label.pack()

# Create the main application window
# Create the main application window
root = tk.Tk()
root.title("Restaurant Analysis")

# Set the size of the main window
window_width = 400
window_height = 250
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Add dropdown menu to select analysis type
analysis_var = tk.StringVar(root)
analysis_var.set("Select Analysis Type")
analysis_label = tk.Label(root, text="Select Analysis Type:")
analysis_label.pack()
analysis_dropdown = tk.OptionMenu(root, analysis_var, "Competitive Analysis", "Individual Analysis")
analysis_dropdown.pack()

# Function to handle selection of analysis type
def select_analysis():
    selected_analysis = analysis_var.get()
    if selected_analysis == "Competitive Analysis":
        # Show restaurant and competitor selection dropdowns
        restaurant_label.pack()
        restaurant_dropdown.pack()
        competitor_label.pack()
        competitor_dropdown.pack()
        perform_button.pack()  # Show perform button for competitive analysis
    elif selected_analysis == "Individual Analysis":
        # Hide competitor selection dropdown
        restaurant_label.pack()
        restaurant_dropdown.pack()
        competitor_label.pack_forget()
        competitor_dropdown.pack_forget()
        perform_button.pack()  # Show perform button for individual analysis

# Add dropdown menu to select restaurant
restaurants = df['Restaurant'].unique()
restaurant_var = tk.StringVar(root)
restaurant_var.set("Select Restaurant")
restaurant_label = tk.Label(root, text="Select Restaurant:")
restaurant_dropdown = tk.OptionMenu(root, restaurant_var, *restaurants)

# Add dropdown menu to select competitor
competitors = df['Restaurant'].unique()
competitors = list(filter(lambda x: x != 'Select Restaurant', competitors))  # Remove "Select Restaurant" option
competitors.insert(0, "Select Competitor")
competitor_var = tk.StringVar(root)
competitor_var.set("Select Competitor")
competitor_label = tk.Label(root, text="Select Competitor:")
competitor_dropdown = tk.OptionMenu(root, competitor_var, *competitors)

# Add button to confirm analysis type selection
confirm_button = tk.Button(root, text="Confirm", command=select_analysis)
confirm_button.pack()

# Add button to perform analysis
perform_button = tk.Button(root, text="Perform Analysis", command=lambda: perform_analysis(analysis_var.get()))

# Function to perform analysis based on selected type
def perform_analysis(selected_analysis):
    if selected_analysis == "Competitive Analysis":
        perform_competitive_analysis()
    elif selected_analysis == "Individual Analysis":
        perform_individual_analysis()
    else:
        tk.messagebox.showinfo("Error", "Please select an analysis type.")

# Start the main event loop
root.mainloop()
