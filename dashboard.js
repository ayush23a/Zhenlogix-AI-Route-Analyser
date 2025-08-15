import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-app.js";
import {
  getAuth,
  signInAnonymously,
  signInWithCustomToken,
  onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/11.6.1/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-firestore.js";

// Global variables for Firebase instances
let app;
let db;
let auth;
let userId = null;
let isAuthReady = false;

// Function to show custom modal
function showModal(title, message) {
  document.getElementById("modal-title").innerText = title;
  document.getElementById("modal-message").innerText = message;
  document.getElementById("custom-modal").classList.remove("hidden");
}

// Function to close custom modal
function closeModal() {
  document.getElementById("custom-modal").classList.add("hidden");
}

// Attach event listener to modal OK button
document
  .getElementById("modal-ok-button")
  .addEventListener("click", closeModal);

// Initialize Firebase and set up auth listener
async function initializeFirebase() {
  try {
    // Access global Firebase config and app ID
    const firebaseConfig =
      typeof __firebase_config !== "undefined"
        ? JSON.parse(__firebase_config)
        : null;
    const appId = typeof __app_id !== "undefined" ? __app_id : "default-app-id";

    if (firebaseConfig) {
      app = initializeApp(firebaseConfig);
      db = getFirestore(app);
      auth = getAuth(app);

      // Sign in with custom token or anonymously
      const initialAuthToken =
        typeof __initial_auth_token !== "undefined"
          ? __initial_auth_token
          : null;

      try {
        if (initialAuthToken) {
          await signInWithCustomToken(auth, initialAuthToken);
        } else {
          await signInAnonymously(auth);
        }
      } catch (error) {
        console.error("Firebase authentication error:", error);
      }

      // Listen for auth state changes
      onAuthStateChanged(auth, (user) => {
        if (user) {
          userId = user.uid;
        } else {
          userId = crypto.randomUUID(); // Use a random ID if not authenticated
        }
        isAuthReady = true;
      });
    } else {
      console.warn("Firebase config not found. Running without Firebase.");
      isAuthReady = true; // Still set auth ready for non-Firebase flow
    }
  } catch (error) {
    console.error("Error initializing Firebase:", error);
    isAuthReady = true; // Ensure app can proceed even if Firebase init fails
  }
}

// Call Firebase initialization
initializeFirebase();

// --- Weather Section Logic ---
const OPENWEATHER_API_KEY =
  ""; // Environment will provide this
let weatherLocation = document.getElementById("weather-location-input").value;

async function fetchWeather(loc) {
  const updateButton = document.getElementById("update-weather-button");
  const weatherErrorDisplay = document.getElementById("weather-error");
  const weatherDataDisplay = document.getElementById("weather-data-display");
  const aiSuggestionDisplay = document.getElementById("ai-suggestion");

  updateButton.innerText = "Updating...";
  updateButton.disabled = true;
  weatherErrorDisplay.classList.add("hidden");
  weatherDataDisplay.classList.add("hidden");
  aiSuggestionDisplay.innerText = "Fetching AI suggestion...";

  if (!OPENWEATHER_API_KEY || OPENWEATHER_API_KEY === "") {
    weatherErrorDisplay.innerText =
      "OpenWeatherMap API key is missing. Please ensure it is correctly provided by the environment.";
    weatherErrorDisplay.classList.remove("hidden");
    updateButton.innerText = "Update";
    updateButton.disabled = false;
    return;
  }

  try {
    // First, get coordinates for the location
    const geoResponse = await fetch(
      `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(
        loc
      )}&limit=1&appid=${OPENWEATHER_API_KEY}`
    );
    if (!geoResponse.ok) {
      throw new Error(`HTTP error! status: ${geoResponse.status}`);
    }
    const geoData = await geoResponse.json();

    if (geoData.length === 0) {
      weatherErrorDisplay.innerText =
        "Location not found. Please try a different city.";
      weatherErrorDisplay.classList.remove("hidden");
      updateButton.innerText = "Update";
      updateButton.disabled = false;
      return;
    }

    const { lat, lon, name, state, country } = geoData[0];
    const formattedLocation = `${name}${state ? ", " + state : ""}${
      country ? ", " + country : ""
    }`;
    document.getElementById("weather-location-input").value = formattedLocation; // Update input with formatted name

    // Then, get current weather using coordinates
    const weatherResponse = await fetch(
      `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${OPENWEATHER_API_KEY}`
    );
    if (!weatherResponse.ok) {
      throw new Error(`HTTP error! status: ${weatherResponse.status}`);
    }
    const data = await weatherResponse.json();

    document.getElementById(
      "weather-city-country"
    ).innerText = `${data.name}, ${data.sys?.country}`;
    document.getElementById(
      "weather-icon"
    ).src = `https://openweathermap.org/img/wn/${data.weather[0].icon}@2x.png`;
    document.getElementById("weather-icon").alt = data.weather[0].description;
    document.getElementById("weather-temp").innerText = data.main?.temp
      ? `${Math.round(data.main.temp)}°C`
      : "N/A";
    document.getElementById("weather-description").innerText =
      data.weather[0]?.description || "N/A";
    document.getElementById(
      "weather-humidity"
    ).innerText = `Humidity: ${data.main?.humidity}%`;
    document.getElementById(
      "weather-cloudiness"
    ).innerText = `Chance of Rain: ${
      data.clouds?.all ? `${data.clouds.all}%` : "N/A"
    } (Cloudiness)`;
    weatherDataDisplay.classList.remove("hidden");

    generateAISuggestion(data); // Generate AI suggestion based on new weather data
  } catch (error) {
    console.error("Error fetching weather data:", error);
    weatherErrorDisplay.innerText = `Failed to fetch weather: ${error.message}. Please check your API key and network connection.`;
    weatherErrorDisplay.classList.remove("hidden");
    weatherDataDisplay.classList.add("hidden");
  } finally {
    updateButton.innerText = "Update";
    updateButton.disabled = false;
  }
}

async function generateAISuggestion(weather) {
  const aiSuggestionDisplay = document.getElementById("ai-suggestion");
  aiSuggestionDisplay.innerText = "Generating AI suggestion...";

  if (!weather) {
    aiSuggestionDisplay.innerText = "No weather data to generate suggestion.";
    return;
  }

  const weatherCondition = weather.weather[0]?.description || "unknown";
  const temperature = weather.main?.temp ? `${weather.main.temp}°C` : "unknown";
  const humidity = weather.main?.humidity
    ? `${weather.main.humidity}%`
    : "unknown";
  const prompt = `Given the weather condition '${weatherCondition}', temperature '${temperature}', and humidity '${humidity}', provide a short, actionable suggestion for a delivery driver. Keep it under 20 words.`;

  try {
    let chatHistory = [];
    chatHistory.push({ role: "user", parts: [{ text: prompt }] });
    const payload = { contents: chatHistory };
    const apiKey = ""; // Gemini API key (empty string, environment will provide)

    if (!apiKey || apiKey === "") {
      aiSuggestionDisplay.innerText =
        "Gemini API key is missing. AI suggestions cannot be generated.";
      return;
    }

    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    if (
      result.candidates &&
      result.candidates.length > 0 &&
      result.candidates[0].content &&
      result.candidates[0].content.parts &&
      result.candidates[0].content.parts.length > 0
    ) {
      const text = result.candidates[0].content.parts[0].text;
      aiSuggestionDisplay.innerText = text;
    } else {
      aiSuggestionDisplay.innerText = "Could not generate AI suggestion.";
      console.error("Unexpected Gemini API response structure:", result);
    }
  } catch (error) {
    console.error("Error generating AI suggestion:", error);
    aiSuggestionDisplay.innerText = "Failed to get AI suggestion.";
  }
}

document
  .getElementById("update-weather-button")
  .addEventListener("click", () => {
    fetchWeather(document.getElementById("weather-location-input").value);
  });

document
  .getElementById("weather-location-input")
  .addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      fetchWeather(document.getElementById("weather-location-input").value);
    }
  });

// Set up interval for weather updates
setInterval(
  () => fetchWeather(document.getElementById("weather-location-input").value),
  300000
); // 5 minutes

// Call fetchWeather on dashboard load
window.addEventListener("load", () => {
  fetchWeather(document.getElementById("weather-location-input").value);
});

// --- Route Planning Logic ---
const destinationsContainer = document.getElementById("destinations-container");
const addDestinationButton = document.getElementById("add-destination-button");
const sourceLocationInput = document.getElementById("source-location");
const navigateButton = document.getElementById("navigate-button");

let destinationCount = 1; // Start with 1 destination input already in HTML

function addDestinationInput() {
  destinationCount++;
  const div = document.createElement("div");
  div.className = "form-group";
  div.id = `destination-group-${destinationCount}`; // Unique ID for the group
  div.innerHTML = `
        <label for="destination-${destinationCount}" class="form-label">
            Destination ${destinationCount}
            <button type="button" class="remove-destination-button" data-destination-id="${destinationCount}">×</button>
        </label>
        <input
            type="text"
            id="destination-${destinationCount}"
            class="form-input destination-input"
            placeholder="e.g., Customer Address ${destinationCount}"
        />
    `;
  destinationsContainer.appendChild(div);

  // Add event listener for the new remove button
  div
    .querySelector(".remove-destination-button")
    .addEventListener("click", function () {
      const idToRemove = this.dataset.destinationId;
      document.getElementById(`destination-group-${idToRemove}`).remove();
      // Re-numbering could be complex, for beginner level, just remove.
      // If strict re-numbering is needed, a more advanced approach would be required.
    });
}

addDestinationButton.addEventListener("click", addDestinationInput);

navigateButton.addEventListener("click", function () {
  const source = sourceLocationInput.value.trim();
  if (!source) {
    showModal("Input Error", "Please enter a Source Location.");
    return;
  }

  const destinationInputs = document.querySelectorAll(".destination-input");
  const destinations = [];
  destinationInputs.forEach((input) => {
    const value = input.value.trim();
    if (value) {
      // Only add non-empty destinations
      destinations.push(value);
    }
  });

  if (destinations.length === 0) {
    showModal("Input Error", "Please enter at least one Destination Location.");
    return;
  }

  const origin = encodeURIComponent(source);
  const destination = encodeURIComponent(destinations[destinations.length - 1]); // Last destination is final
  const waypoints = destinations
    .slice(0, -1)
    .map((d) => encodeURIComponent(d))
    .join("|"); // All except last are waypoints

  let googleMapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${origin}&destination=${destination}`;
  if (waypoints) {
    googleMapsUrl += `&waypoints=${waypoints}`;
  }

  window.open(googleMapsUrl, "_blank");
});

// --- Map Section Logic ---
let map;
let currentMarker;

// Initialize map when the window loads
window.onload = function () {
  // Fix for default marker icon in Leaflet
  if (L.Icon.Default.prototype._getIconUrl) {
    L.Icon.Default.mergeOptions({
      iconRetinaUrl:
        "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
      iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
      shadowUrl:
        "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
    });
  } else {
    L.Icon.Default.imagePath = "https://unpkg.com/leaflet@1.7.1/dist/images/";
  }

  map = L.map("map").setView([22.9868, 87.855], 13); // Default view: West Bengal

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
};

document
  .getElementById("show-current-location-button")
  .addEventListener("click", function () {
    if (!map) return;

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          const latlng = [latitude, longitude];

          // Remove existing marker if any
          if (currentMarker) {
            map.removeLayer(currentMarker);
          }

          // Add new marker
          currentMarker = L.marker(latlng)
            .addTo(map)
            .bindPopup("Your Current Location")
            .openPopup();

          // Center map on current location
          map.setView(latlng, 15);
        },
        (error) => {
          console.error("Error getting current location:", error);
          showModal(
            "Location Error",
            `Unable to retrieve your location: ${error.message}. Please ensure location services are enabled.`
          );
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      );
    } else {
      showModal(
        "Geolocation Not Supported",
        "Your browser does not support geolocation."
      );
    }
  });

document
  .getElementById("go-to-route-planning-button")
  .addEventListener("click", function () {
    const routePlanningSection = document.getElementById(
      "route-planning-section"
    ); // Refers to the route planning card
    if (routePlanningSection) {
      routePlanningSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      // Add a visual cue like a temporary highlight
      routePlanningSection.classList.add("animate-pulse"); // Using generic pulse from global styles
      setTimeout(() => {
        routePlanningSection.classList.remove("animate-pulse");
      }, 1500);
    }
  });

// --- Chatbot Logic ---
const chatbotContainer = document.getElementById("chatbot-container");
const toggleChatbotButton = document.getElementById("toggle-chatbot-button");
const closeChatbotButton = document.getElementById("close-chatbot-button");
const chatWindow = document.getElementById("chat-window");
const chatInput = document.getElementById("chat-input");
const sendChatButton = document.getElementById("send-chat-button");
const chatPlaceholder = document.getElementById("chat-placeholder");

let isChatbotVisible = false;
let loadingChatResponse = false;

function toggleChatbotVisibility() {
  isChatbotVisible = !isChatbotVisible;
  if (isChatbotVisible) {
    chatbotContainer.classList.remove("hidden");
    toggleChatbotButton.innerText = "Hide Chat";
  } else {
    chatbotContainer.classList.add("hidden");
    toggleChatbotButton.innerText = "Chat";
  }
}

async function generateChatResponse(userMessage) {
  loadingChatResponse = true;
  sendChatButton.disabled = true;
  chatInput.disabled = true;

  const typingIndicator = document.createElement("div");
  typingIndicator.className = "text-left margin-bottom-2"; // Added margin-bottom
  typingIndicator.innerHTML =
    '<span class="d-inline-block chatbot-typing-bubble animate-pulse">Typing...</span>';
  chatWindow.appendChild(typingIndicator);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  const prompt = `The user asked: "${userMessage}". Respond as a helpful chatbot for a delivery website, focusing on navigation and features. Keep the response concise and friendly.`;

  try {
    let chatHistory = [];
    chatHistory.push({ role: "user", parts: [{ text: prompt }] });
    const payload = { contents: chatHistory };
    const apiKey = ""; // Gemini API key (empty string, environment will provide)

    if (!apiKey || apiKey === "") {
      const botMessageDiv = document.createElement("div");
      botMessageDiv.className = "text-left margin-bottom-2";
      botMessageDiv.innerHTML =
        '<span class="d-inline-block chatbot-bot-bubble">Chatbot service is unavailable: Gemini API key is missing.</span>';
      chatWindow.removeChild(typingIndicator);
      chatWindow.appendChild(botMessageDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
      return;
    }

    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    const botMessageDiv = document.createElement("div");
    botMessageDiv.className = "text-left margin-bottom-2";

    if (
      result.candidates &&
      result.candidates.length > 0 &&
      result.candidates[0].content &&
      result.candidates[0].content.parts &&
      result.candidates[0].content.parts.length > 0
    ) {
      const text = result.candidates[0].content.parts[0].text;
      botMessageDiv.innerHTML = `<span class="d-inline-block chatbot-bot-bubble">${text}</span>`;
    } else {
      botMessageDiv.innerHTML =
        '<span class="d-inline-block chatbot-bot-bubble">Sorry, I could not generate a response.</span>';
      console.error("Unexpected Gemini API response structure:", result);
    }
    chatWindow.removeChild(typingIndicator);
    chatWindow.appendChild(botMessageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  } catch (error) {
    console.error("Error generating chatbot response:", error);
    const botMessageDiv = document.createElement("div");
    botMessageDiv.className = "text-left margin-bottom-2";
    botMessageDiv.innerHTML =
      '<span class="d-inline-block chatbot-bot-bubble">Failed to connect to the chatbot service.</span>';
    chatWindow.removeChild(typingIndicator);
    chatWindow.appendChild(botMessageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  } finally {
    loadingChatResponse = false;
    sendChatButton.disabled = false;
    chatInput.disabled = false;
  }
}

function handleSendMessage() {
  const userMessage = chatInput.value.trim();
  if (userMessage === "") return;

  if (chatPlaceholder) {
    chatPlaceholder.classList.add("hidden");
  }

  const userMessageDiv = document.createElement("div");
  userMessageDiv.className = "text-right margin-bottom-2"; // Added margin-bottom
  userMessageDiv.innerHTML = `<span class="d-inline-block chatbot-user-bubble">${userMessage}</span>`;
  chatWindow.appendChild(userMessageDiv);
  chatInput.value = "";
  chatWindow.scrollTop = chatWindow.scrollHeight;

  generateChatResponse(userMessage);
}

toggleChatbotButton.addEventListener("click", toggleChatbotVisibility);
closeChatbotButton.addEventListener("click", toggleChatbotVisibility);
sendChatButton.addEventListener("click", handleSendMessage);
chatInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    handleSendMessage();
  }
});

// Add hover effect to chatbot messages (UX improvement)
// (Note: For vanilla CSS, direct inline style manipulation can be done,
// but for more complex effects, CSS classes are preferred)
chatWindow.addEventListener("mouseover", (e) => {
  if (
    e.target.classList.contains("chatbot-user-bubble") ||
    e.target.classList.contains("chatbot-bot-bubble")
  ) {
    e.target.style.backgroundColor = "#e0f2fe"; /* Light blue on hover */
  }
});
chatWindow.addEventListener("mouseout", (e) => {
  if (e.target.classList.contains("chatbot-user-bubble")) {
    e.target.style.backgroundColor =
      ""; /* Reset to original user bubble color */
  } else if (e.target.classList.contains("chatbot-bot-bubble")) {
    e.target.style.backgroundColor =
      ""; /* Reset to original bot bubble color */
  }
});
// dashboard.js (add to the existing script)

// ... (existing imports, global variables, and map initialization)

let currentLocationMarker = null; // Global variable to hold the current location marker

// Function to get and display current location
function getCurrentLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;
        const latlng = [lat, lng];

        // Remove previous marker if it exists
        if (currentLocationMarker) {
          map.removeLayer(currentLocationMarker);
        }

        // Add a new marker for current location
        currentLocationMarker = L.marker(latlng).addTo(map);
        currentLocationMarker.bindPopup("Your Current Location").openPopup();

        // Set map view to current location
        map.setView(latlng, 15); // Zoom level 15 for a good view of the area
      },
      (error) => {
        let errorMessage = "Unable to retrieve your location.";
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = "Location access denied. Please enable location services in your browser settings.";
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage = "Location information is unavailable.";
            break;
          case error.TIMEOUT:
            errorMessage = "The request to get user location timed out.";
            break;
          case error.UNKNOWN_ERROR:
            errorMessage = "An unknown error occurred.";
            break;
        }
        showModal("Location Error", errorMessage);
        console.error("Geolocation error:", error);
      }
    );
  } else {
    showModal("Feature Not Supported", "Geolocation is not supported by your browser.");
  }
}

// Attach event listener to the new current location button
document.addEventListener('DOMContentLoaded', () => {
    // Ensure the button exists before adding the listener
    const currentLocationButton = document.getElementById('current-location-button');
    if (currentLocationButton) {
        currentLocationButton.addEventListener('click', getCurrentLocation);
    }
    // ... (other existing event listeners)
});

// ... (rest of the existing dashboard.js content)
// dashboard.js (add to the existing script)

// ... (existing imports and global variables)

// New global variables for map layers
let currentRouteLayer = null;

// Function to calculate and display route
async function calculateAndDisplayRoute(origin, destination) {
  // Placeholder for a routing API call
  // In a real application, you would integrate with a service like Google Maps Directions API,
  // OpenRouteService, or another mapping provider here.
  // This example simulates a successful API response.
  try {
    // Example API endpoint (replace with actual API)
    // const response = await fetch(`YOUR_ROUTING_API_ENDPOINT?origin=${origin}&destination=${destination}`);
    // const data = await response.json();

    // Simulate a route response
    const mockRouteData = {
      distance: "15.5 km",
      duration: "25 mins",
      legs: [
        { instructions: "Start on Main St and head North." },
        { instructions: "Turn left onto Oak Ave." },
        { instructions: "Arrive at destination." },
      ],
      // Example coordinates for a simple line, replace with actual polyline from API
      polyline: [
        [34.052235, -118.243683], // Los Angeles
        [34.062235, -118.253683],
        [34.072235, -118.263683],
      ],
    };

    // Remove previous route layer if it exists
    if (currentRouteLayer) {
      map.removeLayer(currentRouteLayer);
    }

    // Display route on map (assuming mockRouteData.polyline contains lat/lng pairs)
    currentRouteLayer = L.polyline(mockRouteData.polyline, {
      color: "#3b82f6", // Blue color for the route
      weight: 5,
      opacity: 0.7,
    }).addTo(map);

    // Fit map bounds to the new route
    map.fitBounds(currentRouteLayer.getBounds());

    // Update route details in the UI
    document.getElementById("route-distance").innerText = mockRouteData.distance;
    document.getElementById("route-time").innerText = mockRouteData.duration;
    const directionsDiv = document.getElementById("route-directions");
    directionsDiv.innerHTML = ""; // Clear previous directions
    mockRouteData.legs.forEach((leg, index) => {
      const p = document.createElement("p");
      p.innerText = `${index + 1}. ${leg.instructions}`;
      directionsDiv.appendChild(p);
    });
    document.getElementById("route-details").classList.remove("hidden"); // Show details

  } catch (error) {
    console.error("Error calculating route:", error);
    showModal("Route Error", "Could not calculate route. Please try again.");
    document.getElementById("route-details").classList.add("hidden"); // Hide details on error
  }
}

// Event listener for the optimize route button
document.getElementById("optimize-route-button").addEventListener("click", () => {
  const origin = document.getElementById("origin-input").value;
  const destination = document.getElementById("destination-input").value;

  if (origin && destination) {
    calculateAndDisplayRoute(origin, destination);
  } else {
    showModal("Input Required", "Please enter both origin and destination.");
  }
});

