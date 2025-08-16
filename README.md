
### README: ZhenLogix - Last-Mile Delivery Optimization Model

This project is a full stack web application designed for Walmart Sparkathon, 2025 to address the critical challenges of last-mile delivery. It provides a comprehensive, single-screen dashboard for delivery drivers, integrating intelligent route optimization with real-time data and AI-powered assistance.

---

### Project Overview

The core problem this application solves is the inefficiency inherent in multi-stop delivery routes. By providing a smart, all-in-one tool, we empower drivers to make faster, more informed decisions, which ultimately reduces delivery times and operational costs for the company.

---

### Key Features

* **Intelligent Route Planning**: The application uses algorithms like a **modified Dijkstra's algorithm** to find the shortest path between stops and a **Traveling Salesperson Problem (TSP)** solver to determine the most efficient sequence for multi-destination trips. This is a foundational element for achieving true route optimization.
* **AI-Powered Insights**: The application integrates with the **Gemini API** to provide proactive suggestions based on live weather data, helping drivers prepare for their journey.
* **Interactive Map Interface**: A dynamic map powered by **Leaflet.js** allows drivers to visualize their current location and planned routes, improving situational awareness.
* **Real-time Weather Data**: Fetches up-to-the-minute weather conditions from the **OpenWeatherMap API**, ensuring drivers are always informed.
* **User-Friendly Chatbot**: A conversational chatbot, also powered by the **Gemini API**, offers quick answers to common questions about the application's features and navigation.
* **Carbon Footprint analyzer**: calculates the estimated carbon footprints based on the optimized route provided by the route optimization model.

---

### Technology Stack

* **Frontend**: HTML5, Vanilla CSS3 , and JavaScript.
* **APIs**:
    * **OpenWeatherMap API**: For fetching real-time weather data.
    * **Google Maps API**: For dynamic route generation and navigation.
    * **Gemini API**: For AI-powered suggestions and chatbot functionality.
    * **FastAPI**: For connecting backend with the frontend.
* **Algorithms**:
    * Modified Dijkstra's Algorithm for shortest path calculation.
    * Traveling Salesperson Problem (TSP) algorithm for optimal route sequencing.
* **Backend**: Python

---

### Setup and Installation

1.  **Clone the Repository**:
    ```sh
    git clone https://github.com/ayush23a/Zhenlogix-AI-Route-Analyser.git
    ```

2.  **API Keys**: This project requires API keys for OpenWeatherMap and the Gemini API. You will need to obtain these keys and add them to your environment variables or a configuration file.

3.  **Run the Application**: Simply open the `index.html` file in your web browser. The JavaScript will handle the rest. No local server is required.

---

### Usage

1.  Log in using any credentials on the login page.
2.  On the dashboard, you can enter a source and multiple destination addresses in the "Route Planning" section. The app will generate an optimized route via a Google Maps link.
3.  The "Weather Section" provides real-time weather, along with an AI-generated suggestion for your drive.
4.  Use the "Show Current Location" button on the map to display your position.
5.  Click the "Chat" button to open the chatbot for assistance.
