import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import argparse
import geopandas as gpd
from shapely.geometry import box

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

def load_fonts():
    """Load Roboto fonts from the fonts directory."""
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"[WARNING] Font not found: {path}")
            return None
    return fonts

FONTS = load_fonts()

def generate_output_filename(city, theme_name, output_format, ratio_str):
    """Generate unique output filename."""
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    ratio_slug = ratio_str.replace(':', '-')
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{ratio_slug}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes():
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            themes.append(file[:-5])
    return themes

def load_theme(theme_name="feature_based"):
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if not os.path.exists(theme_file):
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF", "text": "#000000", "gradient_color": "#FFFFFF",
            "water": "#C0C0C0", "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A", "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A", "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A", "road_default": "#3A3A3A"
        }
    with open(theme_file, 'r') as f:
        return json.load(f)

THEME = None

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, :3] = rgb
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start, extent_y_end = 0, 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start, extent_y_end = 0.75, 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def get_attributes_from_gdf(gdf):
    """
    Extracts colors and widths directly from a GeoDataFrame based on highway tags.
    """
    colors = []
    widths = []
    
    # Iterate over the rows of the GeoDataFrame
    for _, row in gdf.iterrows():
        highway = row.get('highway', 'unclassified')
        if isinstance(highway, list): highway = highway[0] if highway else 'unclassified'
        
        # Color Logic
        if highway in ['motorway', 'motorway_link']: color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']: color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']: color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']: color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']: color = THEME['road_residential']
        else: color = THEME['road_default']
        colors.append(color)

        # Width Logic
        if highway in ['motorway', 'motorway_link']: width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']: width = 1.0
        elif highway in ['secondary', 'secondary_link']: width = 0.8
        elif highway in ['tertiary', 'tertiary_link']: width = 0.6
        else: width = 0.4
        widths.append(width)
        
    return colors, widths

def get_coordinates(city, country, max_retries=3):
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)
    query = f"{city}, {country}"
    for attempt in range(max_retries):
        try:
            print(f"  Looking up location (Attempt {attempt+1})...")
            location = geolocator.geocode(query)
            if location:
                print(f"[OK] {location.address}\n[OK] {location.latitude}, {location.longitude}")
                return (location.latitude, location.longitude)
            time.sleep(2**attempt)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2**attempt)
    
    print("Geocoding failed.")
    manual = input("Enter coordinates (lat,lon): ").strip()
    if manual:
        try: return tuple(map(float, manual.split(',')))
        except: pass
    return None

def create_poster(city, country, point, dist, output_file, output_format, ratio_str, offset_x, offset_y, svg_size_mm=None):
    print(f"\nGenerating map for {city}, {country}...")
    plt.rcParams['svg.fonttype'] = 'none'

    # --- 1. CALCULATE DIMENSIONS ---
    # Parse Ratio
    try:
        w_ratio, h_ratio = map(float, ratio_str.split(':'))
        target_aspect = w_ratio / h_ratio
    except ValueError:
        print("Invalid ratio format. Using 2:3")
        target_aspect = 2/3

    # Calculate Viewport in Meters based on 'dist' (Shortest Edge Logic)
    if target_aspect < 1:
        view_width_m = dist * 2
        view_height_m = view_width_m / target_aspect
    else:
        view_height_m = dist * 2
        view_width_m = view_height_m * target_aspect
    
    print(f"[DEBUG] Viewport Geometry: {view_width_m:.0f}m (W) x {view_height_m:.0f}m (H)")

    # Calculate Fetch Distance (Bleed) - 10% buffer PLUS the offset amount
    # We must increase fetch distance to ensure we have data if the user shifts the center
    longest_edge_m = max(view_width_m, view_height_m)
    
    # Base fetch is half the longest edge + 10%
    base_fetch = (longest_edge_m / 2) * 1.1
    
    # Add the maximum offset to the fetch radius to prevent blank edges
    max_offset = max(abs(offset_x), abs(offset_y))
    fetch_dist = base_fetch + max_offset
    
    print(f"[DEBUG] Download Radius: {fetch_dist:.0f}m (Includes offset buffer)")

    # --- 2. DOWNLOAD DATA ---
    with tqdm(total=3, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(point, dist=fetch_dist, dist_type='bbox', network_type='all')
        G = ox.project_graph(G) # Projects to UTM (Meters)
        
        # Convert Graph to GeoDataFrames immediately for clipping
        # We only need edges for the map, nodes are implicitly handled
        _, edges = ox.graph_to_gdfs(G)
        
        pbar.update(1)
        time.sleep(0.5) 
        
        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=fetch_dist)
            if water is not None and not water.empty: water = water.to_crs(G.graph['crs'])
        except: water = None
        pbar.update(1)
        time.sleep(0.3)
        
        pbar.set_description("Downloading parks")
        try:
            parks = ox.features_from_point(point, tags={'leisure': 'park', 'landuse': 'grass'}, dist=fetch_dist)
            if parks is not None and not parks.empty: parks = parks.to_crs(G.graph['crs'])
        except: parks = None
        pbar.update(1)

    # --- 3. DESTRUCTIVE CLIPPING ---
    print("Performing destructive clipping...")
    
    # Calculate Center Point
    # We use the bounds of the downloaded graph as the starting geometric center
    minx, miny, maxx, maxy = edges.total_bounds
    
    # APPLY OFFSET HERE
    # Shift the center of the viewport by the user-requested meters
    x_center = ((minx + maxx) / 2) + offset_x
    y_center = ((miny + maxy) / 2) + offset_y
    
    if offset_x != 0 or offset_y != 0:
        print(f"[DEBUG] Applied Center Offset: X={offset_x}m, Y={offset_y}m")

    # Create the Bounding Box Polygon (The "Cookie Cutter")
    west = x_center - (view_width_m / 2)
    east = x_center + (view_width_m / 2)
    south = y_center - (view_height_m / 2)
    north = y_center + (view_height_m / 2)
    
    clip_box = box(west, south, east, north)
    
    # Clip Roads (Edges)
    clipped_edges = gpd.clip(edges, clip_box)
    
    # Clip Water
    if water is not None and not water.empty:
        clipped_water = gpd.clip(water, clip_box)
        clipped_water = clipped_water[clipped_water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    else:
        clipped_water = None

    # Clip Parks
    if parks is not None and not parks.empty:
        clipped_parks = gpd.clip(parks, clip_box)
        clipped_parks = clipped_parks[clipped_parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    else:
        clipped_parks = None

    print(f"[DEBUG] Clipped Edges: {len(edges)} -> {len(clipped_edges)}")

    # --- 4. SETUP FIGURE ---
    if svg_size_mm:
        longest_edge_inch = svg_size_mm / 25.4
    else:
        longest_edge_inch = 24
    
    if target_aspect < 1: # Portrait
        height_inch = longest_edge_inch
        width_inch = height_inch * target_aspect
    else: # Landscape
        width_inch = longest_edge_inch
        height_inch = width_inch / target_aspect

    if output_format.lower() == 'svg':
        fig, ax = plt.subplots(figsize=(width_inch, height_inch), facecolor=THEME['bg'])
    else:
        fig, ax = plt.subplots(figsize=(width_inch, height_inch), facecolor=THEME['bg'])
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor(THEME['bg'])
    
    # --- 5. PLOT LAYERS (Using Clipped Data) ---
    
    # Plot Water
    if clipped_water is not None and not clipped_water.empty:
        clipped_water.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    
    # Plot Parks
    if clipped_parks is not None and not clipped_parks.empty:
        clipped_parks.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)
    
# Plot Roads
    if not clipped_edges.empty:
        # 1. SEPARATE BRIDGES
        # Check if 'bridge' column exists (it might not if no bridges are in the area)
        if 'bridge' in clipped_edges.columns:
            bridges = clipped_edges[clipped_edges['bridge'] == 'yes']
            roads = clipped_edges[clipped_edges['bridge'] != 'yes']
        else:
            bridges = gpd.GeoDataFrame()
            roads = clipped_edges

        # 2. PLOT STANDARD ROADS (Bottom Layer)
        if not roads.empty:
            r_colors, r_widths = get_attributes_from_gdf(roads)
            roads.plot(ax=ax, color=r_colors, linewidth=r_widths, zorder=3)

        # 3. PLOT BRIDGE CASINGS (The "Eraser" Layer)
        # We draw the bridge wider and in the background color to "cut" through water/roads below
        if not bridges.empty:
            b_colors, b_widths = get_attributes_from_gdf(bridges)
            
            # Make casing 2.5x thicker than the road
            casing_widths = [w * 2.5 for w in b_widths]
            
            # Plot Casing (Background Color)
            bridges.plot(ax=ax, color=THEME['bg'], linewidth=casing_widths, zorder=4)
            
            # 4. PLOT BRIDGE TOPS (The Road Layer)
            # Plot the actual road on top of the casing
            bridges.plot(ax=ax, color=b_colors, linewidth=b_widths, zorder=5)
    
    # --- 6. SET VIEWPORT (Strict Alignment) ---
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect('equal')
    ax.axis('off')
    
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom')
    create_gradient_fade(ax, THEME['gradient_color'], location='top')
    
    # --- 7. TYPOGRAPHY ---
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_sub = FontProperties(fname=FONTS['light'], size=18)
        font_coords = FontProperties(fname=FONTS['regular'], size=12)
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_sub = FontProperties(family='monospace', weight='normal', size=18)
        font_coords = FontProperties(family='monospace', size=12)
        font_attr = FontProperties(family='monospace', size=8)
    
    base_font_size = 60
    safe_char_limit = 7
    
    if len(city) > safe_char_limit:
        scale_factor = safe_char_limit / len(city)
        adjusted_font_size = max(base_font_size * scale_factor, 20)
    else:
        adjusted_font_size = base_font_size
    
    if FONTS:
        font_main_adjusted = FontProperties(fname=FONTS['bold'], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family='monospace', weight='bold', size=adjusted_font_size)

    spaced_city = "  ".join(list(city.upper()))
    
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main_adjusted, zorder=11)
    
    ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)
    
    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0: coords = coords.replace("E", "W")
    
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
    
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
            color=THEME['text'], linewidth=1, zorder=11)

    #ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
    #        color=THEME['text'], alpha=0.5, ha='right', va='bottom', 
    #        fontproperties=font_attr, zorder=11)

    # --- 8. SAVE ---
    print(f"Saving to {output_file}...")
    fmt = output_format.lower()
    
    if fmt == "svg":
        plt.savefig(output_file, format=fmt, facecolor=THEME["bg"], bbox_inches=None, pad_inches=0)
    else:
        plt.savefig(output_file, format=fmt, facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0, dpi=300)

    plt.close()
    print(f"[OK] Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="City Map Poster Generator")
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--theme', '-t', default='feature_based', help='Theme name')
    parser.add_argument('--distance', '-d', type=int, default=15000, help='Map radius (shortest edge) in meters')
    parser.add_argument('--ratio', '-r', default='2:3', help='Aspect ratio (e.g., 2:3, 3:2, 1:1, 16:9)')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'], help='Output format')
    parser.add_argument('--svg-size', type=int, default=300, help='Longest edge size in mm (for SVG)')
    parser.add_argument('--offset-x', type=float, default=0, help='Center offset X in meters (East is +, West is -)')
    parser.add_argument('--offset-y', type=float, default=0, help='Center offset Y in meters (North is +, South is -)')
    parser.add_argument('--lat', type=float, help='Manual Latitude')
    parser.add_argument('--lon', type=float, help='Manual Longitude')
    parser.add_argument('--list-themes', action='store_true')
    
    args = parser.parse_args()
    
    if args.list_themes:
        print("\nAvailable Themes:", ", ".join(get_available_themes()))
        os.sys.exit(0)
    
    if not args.city or not args.country:
        print("Usage: python create_map_poster.py -c <city> -C <country> [options]")
        os.sys.exit(1)
    
    THEME = load_theme(args.theme)
    
    try:
        coords = (args.lat, args.lon) if args.lat and args.lon else get_coordinates(args.city, args.country)
        if coords:
            outfile = generate_output_filename(args.city, args.theme, args.format, args.ratio)
            create_poster(
                args.city, args.country, coords, args.distance, outfile, args.format, 
                args.ratio, args.offset_x, args.offset_y, args.svg_size
            )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()