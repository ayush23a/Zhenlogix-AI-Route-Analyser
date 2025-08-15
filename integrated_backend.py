import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ZhenLogix Integrated Route & Weather API",
    description="Complete backend for route optimization with weather integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RouteRequest(BaseModel):
    source: str
    destinations: List[str]
    algorithm: str = "genetic"
    include_weather: bool = True

class WeatherRequest(BaseModel):
    location: str

class RouteResponse(BaseModel):
    success: bool
    route: List[str]
    total_distance: float
    estimated_time: float
    time_saved: float
    carbon_reduction: float
    efficiency_score: float
    weather_data: Optional[Dict] = None
    message: str

class WeatherResponse(BaseModel):
    success: bool
    location: str
    temperature: float
    description: str
    humidity: int
    cloudiness: int
    wind_speed: float
    pressure: int
    icon: str
    ai_insight: str

# Configuration
WEATHER_API_KEY = ""  # Environment will provide this
GEOCODING_TIMEOUT = 10
WEATHER_CACHE = {}
WEATHER_CACHE_DURATION = 300  # 5 minutes

# Initialize geocoder
geolocator = Nominatim(user_agent="zhenlogix_route_optimizer")

class RouteOptimizer:
    def __init__(self):
        self.population_size = 50
        self.generations = 1000
        self.mutation_rate = 0.01
        self.elite_size = 5
        
    def calculate_distance(self, coord1, coord2):
        """Calculate distance between two coordinates using geodesic distance"""
        return geodesic(coord1, coord2).kilometers
    
    def get_coordinates(self, address):
        """Get coordinates for an address"""
        try:
            location = geolocator.geocode(address, timeout=GEOCODING_TIMEOUT)
            if location:
                return (location.latitude, location.longitude)
            else:
                raise ValueError(f"Could not geocode address: {address}")
        except Exception as e:
            logger.error(f"Error geocoding {address}: {e}")
            raise ValueError(f"Could not geocode address: {address}")
    
    def create_distance_matrix(self, coordinates):
        """Create distance matrix from coordinates"""
        n = len(coordinates)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.calculate_distance(coordinates[i], coordinates[j])
        
        return matrix
    
    def genetic_algorithm(self, distance_matrix):
        """Genetic Algorithm for TSP"""
        n = len(distance_matrix)
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            route = list(range(n))
            random.shuffle(route)
            population.append(route)
        
        best_distance = float('inf')
        best_route = None
        
        for generation in range(self.generations):
            # Calculate fitness
            fitness_scores = []
            for route in population:
                distance = self.calculate_route_distance(route, distance_matrix)
                fitness_scores.append(1 / distance)
            
            # Selection
            elite = self.select_elite(population, fitness_scores)
            new_population = elite.copy()
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            # Track best solution
            for route in population:
                distance = self.calculate_route_distance(route, distance_matrix)
                if distance < best_distance:
                    best_distance = distance
                    best_route = route.copy()
            
            if generation % 100 == 0:
                logger.info(f"Generation {generation}: Best distance = {best_distance:.2f}")
        
        return best_route, best_distance
    
    def simulated_annealing(self, distance_matrix):
        """Simulated Annealing for TSP"""
        n = len(distance_matrix)
        current_route = list(range(n))
        random.shuffle(current_route)
        
        current_distance = self.calculate_route_distance(current_route, distance_matrix)
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = 1000
        cooling_rate = 0.995
        
        for iteration in range(1000):
            # Generate neighbor
            new_route = current_route.copy()
            i, j = random.sample(range(n), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
            
            new_distance = self.calculate_route_distance(new_route, distance_matrix)
            
            # Accept or reject
            delta = new_distance - current_distance
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_route = new_route
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            temperature *= cooling_rate
            
            if iteration % 100 == 0:
                logger.info(f"SA iteration {iteration}: Best distance = {best_distance:.2f}")
        
        return best_route, best_distance
    
    def calculate_route_distance(self, route, distance_matrix):
        """Calculate total distance of a route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += distance_matrix[from_city][to_city]
        return total_distance
    
    def select_elite(self, population, fitness_scores):
        """Select elite individuals"""
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        return [population[i] for i in elite_indices]
    
    def tournament_selection(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 5
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def crossover(self, parent1, parent2):
        """Order crossover"""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        child = [-1] * n
        child[start:end] = parent1[start:end]
        
        remaining = [x for x in parent2 if x not in child[start:end]]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        
        return child
    
    def mutate(self, route):
        """Swap mutation"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

class WeatherService:
    def __init__(self):
        self.api_key = WEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    async def get_weather(self, location: str) -> Optional[Dict]:
        """Get weather data for a location"""
        cache_key = location.lower()
        
        # Check cache
        if cache_key in WEATHER_CACHE:
            cached_data, timestamp = WEATHER_CACHE[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=WEATHER_CACHE_DURATION):
                return cached_data
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}?q={location}&appid={self.api_key}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache the result
                        WEATHER_CACHE[cache_key] = (data, datetime.now())
                        
                        return data
                    else:
                        logger.error(f"Weather API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching weather for {location}: {e}")
            return None
    
    def generate_ai_insight(self, weather_data: Dict) -> str:
        """Generate AI insight based on weather data"""
        temp = weather_data['main']['temp'] - 273.15  # Convert to Celsius
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description'].lower()
        
        insights = []
        
        # Temperature insights
        if temp < 10:
            insights.append("Bundle up, it's chilly!")
        elif temp > 30:
            insights.append("It's quite hot. Seek shade and drink water.")
        else:
            insights.append("Temperature is comfortable for outdoor activities.")
        
        # Humidity insights
        if humidity > 70:
            insights.append("High humidity - expect muggy conditions.")
        elif humidity < 40:
            insights.append("Low humidity - the air might feel dry.")
        else:
            insights.append("Moderate humidity levels.")
        
        # Weather condition insights
        if 'rain' in description:
            insights.append("Don't forget your umbrella!")
        elif 'snow' in description:
            insights.append("Dress warmly and watch for slippery conditions.")
        elif 'cloud' in description:
            insights.append("Cloudy conditions may affect visibility.")
        
        return " ".join(insights)

# Initialize services
route_optimizer = RouteOptimizer()
weather_service = WeatherService()

@app.get("/")
async def root():
    return {
        "message": "ZhenLogix Integrated Route & Weather API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "route_optimization": "/api/optimize-route",
            "weather": "/api/weather",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "route_optimization": "active",
            "weather": "active"
        }
    }

@app.post("/api/optimize-route", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """Optimize route with optional weather integration"""
    try:
        # Get coordinates for all locations
        all_locations = [request.source] + request.destinations
        coordinates = []
        
        for location in all_locations:
            try:
                coords = route_optimizer.get_coordinates(location)
                coordinates.append(coords)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Create distance matrix
        distance_matrix = route_optimizer.create_distance_matrix(coordinates)
        
        # Optimize route
        if request.algorithm.lower() == "genetic":
            best_route, total_distance = route_optimizer.genetic_algorithm(distance_matrix)
        elif request.algorithm.lower() == "simulated_annealing":
            best_route, total_distance = route_optimizer.simulated_annealing(distance_matrix)
        else:
            raise HTTPException(status_code=400, detail="Invalid algorithm. Use 'genetic' or 'simulated_annealing'")
        
        # Convert route indices to location names
        optimized_route = [all_locations[i] for i in best_route]
        
        # Calculate metrics
        avg_speed = 50  # km/h
        estimated_time = total_distance / avg_speed  # hours
        baseline_distance = total_distance * 1.3  # Assume 30% longer without optimization
        time_saved = (baseline_distance - total_distance) / avg_speed
        carbon_reduction = (baseline_distance - total_distance) * 0.2  # kg CO2 saved
        efficiency_score = min(100, max(0, 100 - (total_distance / 100)))  # Higher score for shorter routes
        
        # Get weather data if requested
        weather_data = None
        if request.include_weather:
            weather_data = {}
            for location in optimized_route[:3]:  # Get weather for first 3 locations
                weather = await weather_service.get_weather(location)
                if weather:
                    weather_data[location] = {
                        "temperature": round(weather['main']['temp'] - 273.15, 1),
                        "description": weather['weather'][0]['description'],
                        "humidity": weather['main']['humidity'],
                        "icon": weather['weather'][0]['icon'],
                        "ai_insight": weather_service.generate_ai_insight(weather)
                    }
        
        return RouteResponse(
            success=True,
            route=optimized_route,
            total_distance=round(total_distance, 2),
            estimated_time=round(estimated_time, 2),
            time_saved=round(time_saved, 2),
            carbon_reduction=round(carbon_reduction, 2),
            efficiency_score=round(efficiency_score, 1),
            weather_data=weather_data,
            message="Route optimized successfully"
        )
        
    except Exception as e:
        logger.error(f"Error optimizing route: {e}")
        raise HTTPException(status_code=500, detail=f"Error optimizing route: {str(e)}")

@app.post("/api/weather", response_model=WeatherResponse)
async def get_weather_data(request: WeatherRequest):
    """Get weather data for a specific location"""
    try:
        weather_data = await weather_service.get_weather(request.location)
        
        if not weather_data:
            raise HTTPException(status_code=404, detail=f"Weather data not found for {request.location}")
        
        temp_celsius = weather_data['main']['temp'] - 273.15
        ai_insight = weather_service.generate_ai_insight(weather_data)
        
        return WeatherResponse(
            success=True,
            location=weather_data['name'],
            temperature=round(temp_celsius, 1),
            description=weather_data['weather'][0]['description'],
            humidity=weather_data['main']['humidity'],
            cloudiness=weather_data['clouds']['all'],
            wind_speed=weather_data['wind']['speed'],
            pressure=weather_data['main']['pressure'],
            icon=weather_data['weather'][0]['icon'],
            ai_insight=ai_insight
        )
        
    except Exception as e:
        logger.error(f"Error getting weather for {request.location}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting weather data: {str(e)}")

@app.get("/api/weather/{location}")
async def get_weather_by_location(location: str):
    """Get weather data for a location via GET request"""
    try:
        weather_data = await weather_service.get_weather(location)
        
        if not weather_data:
            raise HTTPException(status_code=404, detail=f"Weather data not found for {location}")
        
        temp_celsius = weather_data['main']['temp'] - 273.15
        ai_insight = weather_service.generate_ai_insight(weather_data)
        
        return {
            "success": True,
            "location": weather_data['name'],
            "temperature": round(temp_celsius, 1),
            "description": weather_data['weather'][0]['description'],
            "humidity": weather_data['main']['humidity'],
            "cloudiness": weather_data['clouds']['all'],
            "wind_speed": weather_data['wind']['speed'],
            "pressure": weather_data['main']['pressure'],
            "icon": weather_data['weather'][0]['icon'],
            "ai_insight": ai_insight
        }
        
    except Exception as e:
        logger.error(f"Error getting weather for {location}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting weather data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print(" Starting ZhenLogix Integrated Route & Weather API Server...")
    print(" API Documentation: http://localhost:8000/docs")
    print(" Health Check: http://localhost:8000/health")
    print(" Weather Endpoint: http://localhost:8000/api/weather/{location}")
    print(" Route Optimization: http://localhost:8000/api/optimize-route")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 