import random
from cow import Cow
from cow_pen import Cow_Pen, Zone
import csv
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Data logs
proximity_tracker = defaultdict(lambda: {"start": None, "last": None, "zone": None})
interactions_log = []
cow_registry_log = []
pen_assignments_log = []

# ----------------------------
# Build Pens
# ----------------------------
def build_pens():
    pens = {}

    # common milking/feeding schedule (6am & 6pm; 7am & 7pm)
    milking_times = [6 * 3600, 18 * 3600]
    feeding_times = [7 * 3600, 19 * 3600]

    # 3 normal pens
    for i in range(1, 4):
        pen = Cow_Pen(id=i, milking_times=milking_times, feeding_times=feeding_times)
        pens[i] = pen
        Cow_Pen.pens[i] = pen

    # 1 dry pen (no milking)
    dry_pen = Cow_Pen(id=Cow_Pen.DRY_PEN_ID, milking_times=[], feeding_times=feeding_times)
    pens[Cow_Pen.DRY_PEN_ID] = dry_pen
    Cow_Pen.pens[Cow_Pen.DRY_PEN_ID] = dry_pen

    # 1 sick pen
    sick_pen = Cow_Pen(id=Cow_Pen.SICK_PEN_ID, milking_times=[], feeding_times=[])
    pens[Cow_Pen.SICK_PEN_ID] = sick_pen
    Cow_Pen.pens[Cow_Pen.SICK_PEN_ID] = sick_pen

    return pens

# ----------------------------
# Populate Cows
# ----------------------------
def build_cows(pens):
    cows = []
    normal_pen_ids = [1, 2, 3]
    dry_pen_id = Cow_Pen.DRY_PEN_ID

    for i in range(20):
        is_dry = random.random() < 0.1
        lactating = not is_dry

        if is_dry:
            pen_id = dry_pen_id
        else:
            # choose a normal pen with available capacity
            available_pens = [pid for pid in normal_pen_ids if len(pens[pid].cows) < 8]
            if not available_pens:
                raise RuntimeError("No available pen for cow placement.")
            pen_id = random.choice(available_pens)
            
        assigned_bunk = random.choice(list(Cow_Pen.pens[pen_id].available_bunks))
        Cow_Pen.pens[pen_id].available_bunks.remove(assigned_bunk)

        cow = Cow(
            cow_id=i,
            tag=f"COW{i}",
            parity=random.randint(1, 4),
            lactating=lactating,
            pen_id=pen_id,
            assigned_resting_zone=Cow_Pen.pens[pen_id].zones[f"resting_bunk_{assigned_bunk}"],
        )
        cows.append(cow)
        # Log cow registry
        cow_registry_log.append([
            cow.id,
            cow.parity,
            "lactating" if cow.lactating else "dry",
            cow.tag,
            ""
        ])

    # initialize friendships AFTER all cows exist
    for cowA in cows:
        for cowB in cows:
            if cowA.id == cowB.id:
                continue
            score = ((cowA.friendliness + cowB.friendliness) / 2.0) * random.uniform(0.5, 1.0)
            cowA.friendships[cowB.id] = score

    return cows

# ----------------------------
# Simulation Loop
# ----------------------------
def run_simulation():
    pens = build_pens()
    cows = build_cows(pens)

    # Log week 0 assignments before the loop
    week = 0
    for cow in cows:
        pen_assignments_log.append([week, cow.id, cow.pen.id])

    total_seconds = 24 * 3600
    for t in tqdm(range(total_seconds), desc="Simulating"):
        for cow in list(cows):  # copy in case sick cows move pens
            cow.update(t)
            
        # After all cows update, but before interaction tracking:
        MAX_INTERACTION_DURATION = 600  # 10 minutes

        # Force-close interactions for cows not in the pen or with pos=None
        current_cow_ids = set(cow.id for cow in cows if cow.pos is not None)
        to_remove = []
        for key, val in proximity_tracker.items():
            cow_i, cow_j = key[1], key[2]
            if cow_i not in current_cow_ids or cow_j not in current_cow_ids:
                start, last, zone_val = val["start"], val["last"], val["zone"]
                if start is not None and last is not None and last - start + 1 >= 60:
                    zone_val = zone_val if (zone_val is not None and str(zone_val).strip() != "") else "free_area"
                    chunk_start = start
                    while chunk_start <= last:
                        chunk_end = min(chunk_start + MAX_INTERACTION_DURATION - 1, last)
                        interactions_log.append([cow_i, cow_j, chunk_start, chunk_end, zone_val])
                        chunk_start += MAX_INTERACTION_DURATION
                to_remove.append(key)
        for key in to_remove:
            proximity_tracker[key] = {"start": None, "last": None, "zone": None}
        
        # Vectorized interaction tracking
        for pen in pens.values():
            cows_in_pen = [cow for cow in cows if cow.pen == pen and cow.pos is not None]
            if len(cows_in_pen) < 2:
                continue
            positions = np.array([cow.pos for cow in cows_in_pen])
            ids = [cow.id for cow in cows_in_pen]
            zones = [pen.get_zone_at(*cow.pos) for cow in cows_in_pen]
            dists = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2))
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    if dists[i, j] <= 150:
                        key = (pen.id, ids[i], ids[j])
                        zone = zones[i]
                        if proximity_tracker[key]["start"] is None:
                            proximity_tracker[key]["start"] = t
                            proximity_tracker[key]["zone"] = zone.name if zone else "free_area"
                        proximity_tracker[key]["last"] = t
                    else:
                        key = (pen.id, ids[i], ids[j])
                        start = proximity_tracker[key]["start"]
                        last = proximity_tracker[key]["last"]
                        if start is not None and last is not None and last - start + 1 >= 60:
                            zone_val = proximity_tracker[key]["zone"]
                            zone_val = zone_val if (zone_val is not None and str(zone_val).strip() != "") else "free_area"
                            chunk_start = start
                            while chunk_start <= last:
                                chunk_end = min(chunk_start + MAX_INTERACTION_DURATION - 1, last)
                                interactions_log.append([
                                    ids[i],
                                    ids[j],
                                    chunk_start,
                                    chunk_end,
                                    zone_val
                                ])
                                chunk_start += MAX_INTERACTION_DURATION
                        proximity_tracker[key] = {"start": None, "last": None, "zone": None}
        
        if t % 86400 == 0:
            day = t // 86400
            print(f"--- End of Day {day} ---")
            
        # Weekly pen assignments (after week 0)
        if t % 604800 == 0 and t != 0:  # every week, but not at t=0
            week = t // 604800
            for cow in cows:
                pen_assignments_log.append([week, cow.id, cow.pen.id])
      
    # Flush ongoing interactions
    for key, val in proximity_tracker.items():
        start, last, zone = val["start"], val["last"], val["zone"]
        if start is not None and last is not None and last - start + 1 >= 60:
            cow_i, cow_j = key[1], key[2]
            zone_val = val["zone"]
            zone_val = zone_val if (zone_val is not None and str(zone_val).strip() != "") else "free_area"
            chunk_start = start
            while chunk_start <= last:
                chunk_end = min(chunk_start + MAX_INTERACTION_DURATION - 1, last)
                interactions_log.append([cow_i, cow_j, chunk_start, chunk_end, zone_val])
                chunk_start += MAX_INTERACTION_DURATION

    # Write CSVs
    with open("interactions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_i", "cow_j", "start_ts", "end_ts", "zone"])
        writer.writerows(interactions_log)

    with open("cow_registry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_id", "parity", "lactation_stage", "tag", "notes"])
        writer.writerows(cow_registry_log)

    with open("pen_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["week", "cow_id", "pen_id"])
        writer.writerows(pen_assignments_log)

    print("Simulation complete.")

if __name__ == "__main__":
    run_simulation()