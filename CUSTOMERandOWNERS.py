import tkinter as tk
from tkinter import messagebox
import pandas as pd

# Sample DataFrame with sentiment analysis results for multiple restaurants
df = pd.read_csv('cleaned_and_analyzed_dataset.csv')  

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
        else:
            return "Other"
    else:
        return "Other"  # Return "Other" for non-string values

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

        # Analyze sentiment based on specific aspects
        aspect_sentiment_analysis_label = tk.Label(results_window, text="Sentiment Analysis by Aspect:")
        aspect_sentiment_analysis_label.pack()

        # Define the aspects to focus on
        aspects = ['Ambience', 'Food Taste', 'Service', 'Price/Value']

        # Calculate sentiment ratios for specific aspects
        for aspect in aspects:
            restaurant_aspect_df = restaurant_df[restaurant_df['aspect_label'] == aspect]
            competitor_aspect_df = competitor_df[competitor_df['aspect_label'] == aspect]
            restaurant_aspect_sentiment_ratio = calculate_sentiment_ratio(restaurant_aspect_df)
            competitor_aspect_sentiment_ratio = calculate_sentiment_ratio(competitor_aspect_df)
            aspect_comparison_text = f"{aspect}:\n{selected_restaurant}: {restaurant_aspect_sentiment_ratio:.2f}\n{selected_competitor}: {competitor_aspect_sentiment_ratio:.2f}"
            aspect_comparison_label = tk.Label(results_window, text=aspect_comparison_text)
            aspect_comparison_label.pack()

# Function to provide restaurant suggestions based on selected aspects
def provide_restaurant_suggestions():
    selected_aspect = aspect_var.get()
    if selected_aspect == "Select Aspect":
        tk.messagebox.showinfo("Error", "Please select an aspect.")
    else:
        suggestions_window = tk.Toplevel(root)
        suggestions_window.title(f"Restaurant Suggestions based on {selected_aspect}")

        # Filter DataFrame for the selected aspect
        filtered_df = df[df['aspect_label'] == selected_aspect]

        # Display restaurant suggestions
        suggestion_text = f"Top 5 Restaurants with the most positive comments for {selected_aspect}:\n"
        top_restaurants = filtered_df.groupby('Restaurant')['predicted_sentiment'].apply(lambda x: (x == 'positive').sum()).nlargest(5)
        for restaurant, count in top_restaurants.items():
            suggestion_text += f"{restaurant}: {count} positive comments\n"
        suggestion_label = tk.Label(suggestions_window, text=suggestion_text)
        suggestion_label.pack()

# Create the main application window
root = tk.Tk()
root.title("Restaurant Suggestions")

# Add dropdown menu to select aspect for restaurant suggestions
aspects = ['Ambience', 'Food Taste', 'Service', 'Price/Value']
aspect_var = tk.StringVar(root)
aspect_var.set("Select Aspect")
aspect_label = tk.Label(root, text="Select Aspect:")
aspect_label.pack()
aspect_dropdown = tk.OptionMenu(root, aspect_var, *aspects)
aspect_dropdown.pack()

# Add button to provide restaurant suggestions based on selected aspect
suggest_button = tk.Button(root, text="Get Restaurant Suggestions", command=provide_restaurant_suggestions)
suggest_button.pack()

# Start the main event loop
root.mainloop()
