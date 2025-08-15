#!/usr/bin/env python3


import math
import random
import json
import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UTILITY FUNCTIONS

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# CONFIGURATION

class Config:
    """Configuration settings"""
    # API Keys
    GOOGLE_MAPS_API_KEY = ''
    HERE_API_KEY = ''
    OPENWEATHER_API_KEY = '4486405d42804787ce9442ec8e964bfc'
    
    # Application Settings
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 8000
    
    # Algorithm Settings
    TSP_ALGORITHM = 'genetic'
    MAX_ITERATIONS = 1000
    POPULATION_SIZE = 50
    
    # Data Settings
    CACHE_DIR = 'cache'
    DATA_DIR = 'data'
    LOGS_DIR = 'logs'
    
    # OSM Settings
    DEFAULT_PLACE = 'Norman, Oklahoma, USA'
    NETWORK_TYPE = 'drive'
    
    # Traffic Settings
    USE_TRAFFIC_DATA = False
    TRAFFIC_UPDATE_INTERVAL = 300

# DATA INTEGRATION

class DataIntegration:
    """Handles integration with various real-world data sources"""
    
    def __init__(self):
        self.config = Config()
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_geocoding_data(self, address: str) -> Optional[Dict]:
        """Get geocoding data from OpenStreetMap Nominatim"""
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': address, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'ZhenLogixRouteOptimizer/1.0'}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            results = response.json()
            if results:
                return {
                    'lat': float(results[0]['lat']),
                    'lon': float(results[0]['lon']),
                    'formatted_address': results[0].get('display_name', address)
                }
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
        return None
    
    def get_weather_data(self, lat: float, lon: float) -> Optional[Dict]:
        """Get weather data from OpenWeatherMap API"""
        if not self.config.OPENWEATHER_API_KEY:
            logger.warning("OpenWeather API key not configured")
            return None
            
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.config.OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'weather_condition': data['weather'][0]['main'],
                'wind_speed': data['wind']['speed'],
                'visibility': data.get('visibility', 10000),
                'humidity': data['main'].get('humidity', 0)
            }
        except Exception as e:
            logger.error(f"Weather data error: {e}")
            
        return None
    
    def get_traffic_data(self, origin: Tuple[float, float], 
                        destination: Tuple[float, float]) -> Optional[Dict]:
        """Get real-time traffic data (mock implementation)"""
        # Mock traffic data for demonstration
        distance = haversine(origin[0], origin[1], destination[0], destination[1])
        base_time = distance * 2  # 2 minutes per km
        traffic_factor = 1.0 + (random.random() * 0.3)  # 0-30% traffic
        
        return {
            'travel_time': base_time * traffic_factor,
            'distance': distance * 1000,  # Convert to meters
            'traffic_delay': base_time * (traffic_factor - 1)
        }
    
    def load_historical_routes(self) -> List[Dict]:
        """Load historical route data (mock implementation)"""
        return [
            {
                'timestamp': datetime.now().isoformat(),
                'total_distance': 45.2,
                'time_saved': 12.3,
                'carbon_reduction': 2.6
            }
        ]

# TSP SOLVER

class EnhancedTSPSolver:
    """Enhanced TSP solver with multiple algorithms"""
    
    def __init__(self, algorithm: str = 'genetic'):
        self.config = Config()
        self.algorithm = algorithm
        self.best_solution = None
        self.best_distance = float('inf')
        
    def solve(self, locations: List[Dict]) -> Dict:
        """Solve TSP using the specified algorithm"""
        if not locations:
            return {'order': [], 'distance': 0, 'algorithm': self.algorithm}
        
        if len(locations) == 1:
            return {
                'order': [locations[0]['id']], 
                'distance': 0, 
                'algorithm': self.algorithm
            }
        
        if self.algorithm == 'genetic':
            return self._genetic_algorithm(locations)
        elif self.algorithm == 'simulated_annealing':
            return self._simulated_annealing(locations)
        elif self.algorithm == 'nearest_neighbor':
            return self._nearest_neighbor(locations)
        else:
            logger.warning(f"Unknown algorithm: {self.algorithm}, using nearest neighbor")
            return self._nearest_neighbor(locations)
    
    def _calculate_total_distance(self, order: List[int], locations: List[Dict]) -> float:
        """Calculate total distance for a given order"""
        total_distance = 0
        for i in range(len(order)):
            current = locations[order[i]]
            next_idx = order[(i + 1) % len(order)]
            next_loc = locations[next_idx]
            
            distance = haversine(
                current['lat'], current['lon'],
                next_loc['lat'], next_loc['lon']
            )
            total_distance += distance
        
        return total_distance
    
    def _nearest_neighbor(self, locations: List[Dict]) -> Dict:
        """Nearest neighbor algorithm"""
        n = len(locations)
        visited = [False] * n
        order = [0]
        visited[0] = True
        
        for _ in range(1, n):
            last = order[-1]
            nearest = 0
            min_dist = float('inf')
            
            for i in range(n):
                if not visited[i]:
                    d = haversine(
                        locations[last]['lat'], locations[last]['lon'],
                        locations[i]['lat'], locations[i]['lon']
                    )
                    if d < min_dist:
                        min_dist = d
                        nearest = i
            
            order.append(nearest)
            visited[nearest] = True
        
        total_distance = self._calculate_total_distance(order, locations)
        
        return {
            'order': [locations[i]['id'] for i in order],
            'distance': total_distance,
            'algorithm': 'nearest_neighbor'
        }
    
    def _genetic_algorithm(self, locations: List[Dict]) -> Dict:
        """Genetic algorithm for TSP"""
        n = len(locations)
        population_size = self.config.POPULATION_SIZE
        max_iterations = self.config.MAX_ITERATIONS
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = list(range(n))
            random.shuffle(individual)
            population.append(individual)
        
        best_individual = None
        best_distance = float('inf')
        
        for generation in range(max_iterations):
            # Calculate fitness for each individual
            fitness_scores = []
            for individual in population:
                distance = self._calculate_total_distance(individual, locations)
                fitness_scores.append(1 / (distance + 1))  # Avoid division by zero
                
                if distance < best_distance:
                    best_distance = distance
                    best_individual = individual.copy()
            
            # Selection
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament = random.sample(range(population_size), tournament_size)
                winner = max(tournament, key=lambda i: fitness_scores[i])
                new_population.append(population[winner].copy())
            
            # Crossover and mutation
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    # Crossover
                    if random.random() < 0.8:  # 80% crossover rate
                        child1, child2 = self._crossover(
                            new_population[i], new_population[i + 1]
                        )
                        new_population[i] = child1
                        new_population[i + 1] = child2
                    
                    # Mutation
                    if random.random() < 0.1:  # 10% mutation rate
                        self._mutate(new_population[i])
                    if random.random() < 0.1:
                        self._mutate(new_population[i + 1])
            
            population = new_population
            
            if generation % 100 == 0:
                logger.info(f"Generation {generation}: Best distance = {best_distance:.2f}")
        
        if best_individual is None:
            best_individual = list(range(n))
            
        return {
            'order': [locations[i]['id'] for i in best_individual],
            'distance': best_distance,
            'algorithm': 'genetic'
        }
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) for TSP"""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        # Create child1
        child1 = [-1] * n
        child1[start:end] = parent1[start:end]
        
        remaining = [x for x in parent2 if x not in child1[start:end]]
        j = 0
        for i in range(n):
            if child1[i] == -1:
                child1[i] = remaining[j]
                j += 1
        
        # Create child2
        child2 = [-1] * n
        child2[start:end] = parent2[start:end]
        
        remaining = [x for x in parent1 if x not in child2[start:end]]
        j = 0
        for i in range(n):
            if child2[i] == -1:
                child2[i] = remaining[j]
                j += 1
        
        return child1, child2
    
    def _mutate(self, individual: List[int]):
        """Swap mutation for TSP"""
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    
    def _simulated_annealing(self, locations: List[Dict]) -> Dict:
        """Simulated annealing for TSP"""
        n = len(locations)
        current_order = list(range(n))
        random.shuffle(current_order)
        
        current_distance = self._calculate_total_distance(current_order, locations)
        best_order = current_order.copy()
        best_distance = current_distance
        
        temperature = 1000
        cooling_rate = 0.995
        
        for iteration in range(1000):
            # Generate neighbor
            i, j = random.sample(range(n), 2)
            neighbor_order = current_order.copy()
            neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
            
            neighbor_distance = self._calculate_total_distance(neighbor_order, locations)
            
            # Accept or reject
            delta = neighbor_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_order = neighbor_order
                current_distance = neighbor_distance
                
                if current_distance < best_distance:
                    best_order = current_order.copy()
                    best_distance = current_distance
            
            temperature *= cooling_rate
            
            if iteration % 100 == 0:
                logger.info(f"SA iteration {iteration}: Best distance = {best_distance:.2f}")
        
        return {
            'order': [locations[i]['id'] for i in best_order],
            'distance': best_distance,
            'algorithm': 'simulated_annealing'
        }
    
    def compare_algorithms(self, locations: List[Dict]) -> Dict:
        """Compare different TSP algorithms"""
        algorithms = ['nearest_neighbor', 'genetic', 'simulated_annealing']
        results = {}
        
        for algorithm in algorithms:
            self.algorithm = algorithm
            results[algorithm] = self.solve(locations)
        
        return results

# ============================================================================
# ROUTING ENGINE
# ============================================================================

class EnhancedRouting:
    """Enhanced routing with traffic awareness and multiple modes"""
    
    def __init__(self):
        self.config = Config()
        self.data_integration = DataIntegration()
        self.graphs = {}  # Cache for different places and modes
        
    def find_optimal_route(self, start_coords: Tuple[float, float], 
                          end_coords: Tuple[float, float], 
                          use_traffic: bool = True) -> Dict:
        """Find optimal route between two points"""
        try:
            # Calculate direct distance
            distance = haversine(start_coords[0], start_coords[1], 
                               end_coords[0], end_coords[1])
            
            # Get traffic data if requested
            traffic_data = None
            if use_traffic:
                traffic_data = self.data_integration.get_traffic_data(start_coords, end_coords)
            
            # Calculate travel time
            if traffic_data:
                travel_time = traffic_data['travel_time'] / 60  # Convert to minutes
            else:
                travel_time = distance / 30 * 60  # Assume 30 km/h average speed
            
            return {
                'start_coords': start_coords,
                'end_coords': end_coords,
                'distance': distance,
                'travel_time': travel_time,
                'traffic_data': traffic_data
            }
            
        except Exception as e:
            logger.error(f"Error finding route: {e}")
            return {
                'error': str(e)
            }
    
    def get_route_statistics(self, route_data: Dict) -> Dict:
        """Get statistics for a route"""
        if 'error' in route_data:
            return {'error': route_data['error']}
        
        distance = route_data.get('distance', 0)
        travel_time = route_data.get('travel_time', 0)
        
        return {
            'total_distance_km': distance,
            'estimated_time_minutes': travel_time,
            'average_speed_kmh': (distance / travel_time * 60) if travel_time > 0 else 0
        }

# ============================================================================
# MAIN OPTIMIZER
# ============================================================================

class EnhancedRouteOptimizer:
    """Main class that orchestrates the enhanced route optimization system"""
    
    def __init__(self):
        self.config = Config()
        self.data_integration = DataIntegration()
        self.tsp_solver = EnhancedTSPSolver()
        self.routing = EnhancedRouting()
    
    def optimize_route(self, locations: List[Dict], algorithm: str = 'genetic') -> Dict:
        """Optimize route for given locations"""
        if len(locations) < 2:
            return {
                'order': [loc['id'] for loc in locations],
                'distance': 0,
                'estimated_time': 0,
                'time_saved': 0,
                'carbon_reduction': 0,
                'efficiency_score': 100
            }
        
        # Solve TSP
        self.tsp_solver.algorithm = algorithm
        tsp_result = self.tsp_solver.solve(locations)
        
        # Calculate statistics
        total_distance = tsp_result['distance']
        estimated_time = total_distance / 30 * 60  # Convert to minutes
        
        # Calculate random distance for comparison
        random_distance = self._calculate_random_distance(locations)
        time_saved = max(0, (random_distance - total_distance) / 30 * 60)
        carbon_reduction = max(0, (random_distance - total_distance) * 0.2)
        efficiency_score = min(100, max(0, (1 - (total_distance / random_distance)) * 100))
        
        return {
            'order': tsp_result['order'],
            'distance': round(total_distance, 2),
            'estimated_time': round(estimated_time, 1),
            'time_saved': round(time_saved, 1),
            'carbon_reduction': round(carbon_reduction, 2),
            'efficiency_score': round(efficiency_score, 1),
            'algorithm': algorithm
        }
    
    def _calculate_random_distance(self, locations: List[Dict]) -> float:
        """Calculate distance for random order"""
        random_order = list(range(len(locations)))
        random.shuffle(random_order)
        
        total_distance = 0
        for i in range(len(random_order)):
            current = locations[random_order[i]]
            next_idx = random_order[(i + 1) % len(random_order)]
            next_loc = locations[next_idx]
            
            distance = haversine(
                current['lat'], current['lon'],
                next_loc['lat'], next_loc['lon']
            )
            total_distance += distance
        
        return total_distance

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="ZhenLogix Route Optimization API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
optimizer = EnhancedRouteOptimizer()
data_integration = DataIntegration()

# Pydantic models
class LocationInput(BaseModel):
    address: str
    lat: Optional[float] = None
    lon: Optional[float] = None

class RouteRequest(BaseModel):
    source: LocationInput
    destinations: List[LocationInput]
    algorithm: str = "genetic"
    use_traffic: bool = True

class WeatherRequest(BaseModel):
    lat: float
    lon: float
    location_name: Optional[str] = None

class RouteResponse(BaseModel):
    route_order: List[str]
    total_distance: float
    estimated_time: float
    time_saved: float
    carbon_reduction: float
    efficiency_score: float
    route_path: List[Dict]
    weather_data: Optional[Dict] = None

# API Endpoints
@app.get("/")
async def root():
    return {"message": "ZhenLogix Route Optimization API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/optimize-route", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """Optimize route using TSP algorithms and routing"""
    try:
        # Convert addresses to coordinates if needed
        locations = []
        
        # Process source location
        source_coords = await get_coordinates(request.source)
        locations.append({
            'id': 'source',
            'name': request.source.address,
            'lat': source_coords['lat'],
            'lon': source_coords['lon']
        })
        
        # Process destination locations
        for i, dest in enumerate(request.destinations):
            dest_coords = await get_coordinates(dest)
            locations.append({
                'id': f'dest_{i+1}',
                'name': dest.address,
                'lat': dest_coords['lat'],
                'lon': dest_coords['lon']
            })
        
        # Optimize route
        result = optimizer.optimize_route(locations, request.algorithm)
        
        # Get weather data for source location
        weather_data = None
        try:
            weather_data = data_integration.get_weather_data(
                source_coords['lat'], 
                source_coords['lon']
            )
        except Exception as e:
            logger.warning(f"Could not fetch weather data: {e}")
        
        # Create route path for visualization
        route_path = []
        for loc_id in result['order']:
            loc = next((l for l in locations if l['id'] == loc_id), None)
            if loc:
                route_path.append({
                    'id': loc['id'],
                    'name': loc['name'],
                    'lat': loc['lat'],
                    'lon': loc['lon']
                })
        
        return RouteResponse(
            route_order=result['order'],
            total_distance=result['distance'],
            estimated_time=result['estimated_time'],
            time_saved=result['time_saved'],
            carbon_reduction=result['carbon_reduction'],
            efficiency_score=result['efficiency_score'],
            route_path=route_path,
            weather_data=weather_data
        )
        
    except Exception as e:
        logger.error(f"Error optimizing route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/weather")
async def get_weather_data(request: WeatherRequest):
    """Get weather data for a location"""
    try:
        weather_data = data_integration.get_weather_data(request.lat, request.lon)
        
        if not weather_data:
            raise HTTPException(status_code=404, detail="Weather data not available")
        
        return {
            "weather": weather_data,
            "location": {
                "lat": request.lat,
                "lon": request.lon,
                "name": request.location_name
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/route-statistics")
async def get_route_statistics():
    """Get historical route statistics"""
    try:
        historical_routes = data_integration.load_historical_routes()
        
        if not historical_routes:
            return {
                "total_routes": 0,
                "average_distance": 0,
                "average_time_saved": 0,
                "total_carbon_reduction": 0
            }
        
        total_routes = len(historical_routes)
        total_distance = sum(route.get('total_distance', 0) for route in historical_routes)
        total_time_saved = sum(route.get('time_saved', 0) for route in historical_routes)
        total_carbon = sum(route.get('carbon_reduction', 0) for route in historical_routes)
        
        return {
            "total_routes": total_routes,
            "average_distance": round(total_distance / total_routes, 2) if total_routes > 0 else 0,
            "average_time_saved": round(total_time_saved / total_routes, 2) if total_routes > 0 else 0,
            "total_carbon_reduction": round(total_carbon, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting route statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_coordinates(location: LocationInput) -> Dict[str, float]:
    """Get coordinates for a location"""
    if location.lat and location.lon:
        return {'lat': location.lat, 'lon': location.lon}
    
    # Geocode the address
    geocoded = data_integration.get_geocoding_data(location.address)
    if not geocoded:
        raise HTTPException(status_code=400, detail=f"Could not geocode address: {location.address}")
    
    return {'lat': geocoded['lat'], 'lon': geocoded['lon']}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_demo():
    """Run a demonstration of the route optimization system"""
    print("🚀 Starting ZhenLogix Route Optimization Demo")
    
    # Sample locations
    locations = [
        {'id': 'source', 'name': 'Walmart Distribution Center', 'lat': 35.4676, 'lon': -97.5164},
        {'id': 'dest_1', 'name': 'Customer Address 1', 'lat': 35.2226, 'lon': -97.4395},
        {'id': 'dest_2', 'name': 'Customer Address 2', 'lat': 35.6528, 'lon': -97.4781},
    ]
    
    # Test different algorithms
    algorithms = ['nearest_neighbor', 'genetic', 'simulated_annealing']
    
    print("\n" + "="*60)
    print("TSP ALGORITHM COMPARISON")
    print("="*60)
    
    best_algorithm = None
    best_distance = float('inf')
    
    for algorithm in algorithms:
        result = optimizer.optimize_route(locations, algorithm)
        distance = result['distance']
        print(f"{algorithm.upper():<20} | Distance: {distance:>8.2f} km | Time Saved: {result['time_saved']:>6.1f} min")
        
        if distance < best_distance:
            best_distance = distance
            best_algorithm = algorithm
    
    if best_algorithm:
        print(f"\n🏆 Best Algorithm: {best_algorithm.upper()} ({best_distance:.2f} km)")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        run_demo()
    else:
        print("🚀 Starting ZhenLogix Route Optimization API Server...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🌐 Health Check: http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000) 