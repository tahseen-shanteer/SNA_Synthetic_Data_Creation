import numpy as np
from astar import AStar
import random
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Zone():
    def __init__(self, name, corners, entrance=None):
        self.name = name
        self.corners = corners
        self.center = ((corners[0][0] + corners[1][0]) // 2, (corners[0][1] + corners[1][1]) // 2)
        self.entrance = entrance if entrance in ['N', 'S', 'E', 'W'] else None
        self._calculate_interior_bounds()

    def _calculate_interior_bounds(self):
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        self.left = min(x1, x2)
        self.right = max(x1, x2)
        self.top = min(y1, y2)
        self.bottom = max(y1, y2)
        self.interior_left = self.left + 1
        self.interior_right = self.right - 1
        self.interior_top = self.top + 1
        self.interior_bottom = self.bottom - 1
        self.has_interior = (
            self.interior_left <= self.interior_right and
            self.interior_top <= self.interior_bottom
        )
    
    def get_random_point(self, avoid_borders=True):
        if avoid_borders:
            if not self.has_interior:
                return None
            x = random.randint(self.interior_left, self.interior_right)
            y = random.randint(self.interior_top, self.interior_bottom)
        else:
            x = random.randint(self.left, self.right)
            y = random.randint(self.top, self.bottom)
        return (x, y)
    
class Cow_Pen():
    pens = {}   # mapping id -> Cow_Pen instance
    available_feeding_windows = {}
    
    DRY_PEN_ID = -1
    SICK_PEN_ID = -2

    def __init__(self, id, milking_times, feeding_times):
        self.id = id
        self.width = 976
        self.height = 1220
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.zones = {}
        self.milking_times = milking_times
        self.feeding_times = feeding_times
        self.total_distance_to_and_from_milking = 8900 # cm 
        self.feeding_window_assignments = {}
        self.available_bunks = set(range(1, 9))
        self.cows = []
        self.free_space_points = None
        self.zone_grid = None
        self.construct_pen()
        Cow_Pen.pens[self.id] = self
        
    def register_cow(self, cow):   
        self.cows.append(cow)
    
    def _build_zone_grid(self):
        """Precompute a grid mapping each (x, y) to a zone name or None."""
        self.zone_grid = np.full((self.height, self.width), None, dtype=object)
        for zone in self.zones.values():
            x1, y1 = zone.corners[0]
            x2, y2 = zone.corners[1]
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            self.zone_grid[top:bottom+1, left:right+1] = zone.name

    def _precompute_free_space(self):
        """Precompute all coordinates that are not in any zone (vectorized)."""
        if self.zone_grid is None:
            self._build_zone_grid()
        mask = np.equal(self.zone_grid, None)
        ys, xs = np.where(mask)
        self.free_space_points = list(zip(xs, ys))
        print(f"Precomputed {len(self.free_space_points)} free space points")

    def get_random_point_in_free_area(self):
        if not self.free_space_points:
            self._precompute_free_space()
            if not self.free_space_points:
                return None
        return random.choice(self.free_space_points)
        
    def construct_pen(self):
        # 8 feeding windows
        for i in range(8):
            feeding_zone = Zone(
                name=f"feeding_window_{i+1}",
                corners=((122*i, 0), (122*(i+1) - 1, 61)),
                entrance='S'
            )
            self.add_zone(feeding_zone)
            
        # 4 resting beds on the left, 4 on the right
        for i in range(2):
            for j in range(2):
                zone = Zone(
                    name=f"resting_bunk_{i*2 + j + 1}",
                    corners=((122*j, 366 + 244*i), (122*(j+1), 610 + 244*i)),
                    entrance='N' if i == 0 else 'S'
                )
                self.add_zone(zone)
        for i in range(2):
            for j in range(2):
                zone = Zone(
                    name=f"resting_bunk_{i*2 + j + 5}",
                    corners=((732 + 122*j, 366 + 244*i), (732 + 122*(j+1) - 1, 610 + 244*i)),
                    entrance='N' if i == 0 else 'S'
                )
                self.add_zone(zone)
    
        # water trough
        water_trough = Zone(
            name="water_trough",
            corners=((244, 488), (305, 732)),
            entrance='E'
        )
        self.add_zone(water_trough)
        
        # entrance/exit
        entrance = Zone(
            name="entrance",
            corners=((732, 1220 - 61), (976 - 1, 1220 - 1)),
            entrance='N'
        )
        self.add_zone(entrance)
        
        self._build_zone_grid()
        self._precompute_free_space()
    
    def get_unoccupied_feeding_windows(self):
        return [
            zone for name, zone in self.zones.items()
            if name.startswith("feeding_window") and name not in self.feeding_window_assignments
        ]

    def assign_feeding_window(self, cow):
        available = self.get_unoccupied_feeding_windows()
        if not available:
            return None
        zone = random.choice(available)
        self.feeding_window_assignments[zone.name] = cow.id
        return zone

    def unassign_feeding_window(self, cow):
        for name, assigned_id in list(self.feeding_window_assignments.items()):
            if assigned_id == cow.id:
                del self.feeding_window_assignments[name]
                break

    def zones_overlap(self, zone1_corners, zone2_corners):
        (x1, y1), (x2, y2) = zone1_corners
        (x3, y3), (x4, y4) = zone2_corners
        left1, right1 = min(x1, x2), max(x1, x2)
        top1, bottom1 = min(y1, y2), max(y1, y2)
        left2, right2 = min(x3, x4), max(x3, x4)
        top2, bottom2 = min(y3, y4), max(y3, y4)
        if (right1 <= left2 or left1 >= right2 or 
            bottom1 <= top2 or top1 >= bottom2):
            return False
        return True

    def zone_within_bounds(self, corners):
        (x1, y1), (x2, y2) = corners
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        return (left >= 0 and right < self.width and 
                top >= 0 and bottom < self.height)

    def add_zone(self, zone: Zone):
        if not self.zone_within_bounds(zone.corners):
            print(f"Cannot add {zone.name}: Zone is outside grid boundaries")
            return False
        for existing_zone_name, existing_zone in self.zones.items():
            if self.zones_overlap(zone.corners, existing_zone.corners):
                print(f"Cannot add {zone.name}: Overlaps with existing zone {existing_zone_name}")
                return False
        self.zones[zone.name] = zone
        x1, y1 = zone.corners[0]
        x2, y2 = zone.corners[1]
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        # Only draw borders if NOT entrance
        if zone.name != "entrance":
            self.grid[top:bottom+1, left] = 1
            self.grid[top:bottom+1, right] = 1
            self.grid[top, left:right+1] = 1
            self.grid[bottom, left:right+1] = 1
            if zone.entrance:
                if zone.entrance == 'N':
                    entrance_start = left + 1
                    entrance_end = right
                    self.grid[top, entrance_start:entrance_end] = 0
                elif zone.entrance == 'S':
                    entrance_start = left + 1
                    entrance_end = right
                    self.grid[bottom, entrance_start:entrance_end] = 0
                elif zone.entrance == 'E':
                    entrance_start = top + 1
                    entrance_end = bottom
                    self.grid[entrance_start:entrance_end, right] = 0
                elif zone.entrance == 'W':
                    entrance_start = top + 1
                    entrance_end = bottom
                    self.grid[entrance_start:entrance_end, left] = 0
        print(f"Successfully added {zone.name}")
        return True

    def get_zone_by_name(self, name: str) -> Optional[Zone]:
        return self.zones.get(name)

    def get_zone_at(self, x: int, y: int) -> Optional[Zone]:
        if self.zone_grid is not None:
            name = self.zone_grid[y, x]
            return self.zones.get(name) if name else None
        # fallback (should not happen)
        for zone in self.zones.values():
            x1, y1 = zone.corners[0]
            x2, y2 = zone.corners[1]
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            if (left <= x <= right) and (top <= y <= bottom):
                return zone
        return None

    def get_path_to_zone_center(self, start, zone: Zone):
        astar = AStar(self.grid)
        # Convert (x, y) to (y, x)
        start_yx = (start[1], start[0])
        goal_yx = (int(zone.center[1]), int(zone.center[0]))
        path = astar.find_path(start_yx, goal_yx)
        # Convert back to (x, y)
        if path:
            path = [(p[1], p[0]) for p in path]
        return path

    def get_path_to_random_point_in_zone(self, start, zone: Zone):
        astar = AStar(self.grid)
        goal = zone.get_random_point()
        max_tries = 50
        tries = 0
        while (goal is None or self.grid[goal[1], goal[0]] == 1) and tries < max_tries:
            goal = zone.get_random_point()
            tries += 1
        if goal is None or self.grid[goal[1], goal[0]] == 1:
            return []
        # Convert (x, y) to (y, x)
        start_yx = (start[1], start[0])
        goal_yx = (goal[1], goal[0])
        path = astar.find_path(start_yx, goal_yx)
        # Convert back to (x, y)
        if path:
            path = [(p[1], p[0]) for p in path]
        return path

    def get_path_to_point(self, start, goal):
        astar = AStar(self.grid)
        # Convert (x, y) to (y, x)
        start_yx = (start[1], start[0])
        goal_yx = (goal[1], goal[0])
        path = astar.find_path(start_yx, goal_yx)
        # Convert back to (x, y)
        if path:
            path = [(p[1], p[0]) for p in path]
        return path

    def print_grid(self):
        print("\nGrid visualization:")
        print("   " + " ".join(str(i % 10) for i in range(self.width)))
        print("  " + "--" * self.width)
        for i, row in enumerate(self.grid):
            row_str = " ".join('█' if x == 1 else '·' for x in row)
            print(f"{i:2}|{row_str}")
        print(f"\nZones: {[zone.name for name, zone in self.zones.items()]}")
    
    def plot_grid(self, show_zones=True, show=True, cows=None):
        """
        Visualize the pen grid and zones using matplotlib.
        Optionally, plot cow positions if a list of cows is provided.
        Entrances are shown as yellow lines.
        """
        fig, ax = plt.subplots(figsize=(12, 15))
        ax.imshow(self.grid, cmap='Greys', origin='upper')

        if show_zones:
            for zone in self.zones.values():
                x1, y1 = zone.corners[0]
                x2, y2 = zone.corners[1]
                left, right = min(x1, x2), max(x1, x2)
                top, bottom = min(y1, y2), max(y1, y2)
                rect = patches.Rectangle(
                    (left, top), right - left + 1, bottom - top + 1,
                    linewidth=2, edgecolor='red', facecolor='none', alpha=0.5
                )
                ax.add_patch(rect)
                ax.text(
                    (left + right) / 2, (top + bottom) / 2, zone.name,
                    color='blue', fontsize=8, ha='center', va='center', alpha=0.7
                )
                # Draw entrance if present
                if zone.entrance:
                    if zone.entrance == 'N':
                        ax.plot(
                            [left + 1, right - 1],
                            [top, top],
                            color='yellow', linewidth=4, solid_capstyle='round'
                        )
                    elif zone.entrance == 'S':
                        ax.plot(
                            [left + 1, right - 1],
                            [bottom, bottom],
                            color='yellow', linewidth=4, solid_capstyle='round'
                        )
                    elif zone.entrance == 'E':
                        ax.plot(
                            [right, right],
                            [top + 1, bottom - 1],
                            color='yellow', linewidth=4, solid_capstyle='round'
                        )
                    elif zone.entrance == 'W':
                        ax.plot(
                            [left, left],
                            [top + 1, bottom - 1],
                            color='yellow', linewidth=4, solid_capstyle='round'
                        )

        if cows is not None:
            for cow in cows:
                if cow.pos is not None:
                    ax.plot(cow.pos[0], cow.pos[1], 'go', markersize=6)
                    ax.text(cow.pos[0], cow.pos[1], str(cow.id), color='green', fontsize=7)

        ax.set_title(f"Pen {self.id} Grid Visualization")
        ax.set_xlabel("X (columns)")
        ax.set_ylabel("Y (rows)")
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        plt.tight_layout()
        if show:
            plt.show()
        return ax
