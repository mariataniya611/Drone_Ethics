import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np

# =====================================
# LOAD YOLO MODEL
# =====================================

model = YOLO("yolov8n.pt")

# =====================================
# PACKAGE CONFIGURATION
# =====================================

PACKAGE_TYPES = {
    "1": {
        "name": "Medicine",
        "priority": "CRITICAL",
        "priority_color": "#ff4444",
        "damage_multiplier": 1.5,   # More sensitive = higher damage risk
        "description": "Life-saving medication — handle with extreme care",
    },
    "2": {
        "name": "Electronics",
        "priority": "HIGH",
        "priority_color": "#ff8800",
        "damage_multiplier": 1.3,
        "description": "Fragile circuits — avoid hard surfaces",
    },
    "3": {
        "name": "Documents",
        "priority": "MEDIUM",
        "priority_color": "#ffdd00",
        "damage_multiplier": 0.9,
        "description": "Paper documents — moisture risk on wet grass",
    },
    "4": {
        "name": "Food",
        "priority": "LOW",
        "priority_color": "#44cc77",
        "damage_multiplier": 1.0,
        "description": "Standard perishable delivery",
    },
}

# =====================================
# LANDING ZONE RANKING SYSTEM
# (Feature 6)
# =====================================

SURFACE_SCORES = {
    "Grass / Open Ground":  100,
    "Dirt / Soil":           90,
    "Concrete / Asphalt":    70,
    "Unknown":               40,
}

# Base damage risk by surface (%)
SURFACE_DAMAGE_RISK = {
    "Grass / Open Ground":   5,
    "Dirt / Soil":          10,
    "Concrete / Asphalt":   60,
    "Unknown":              40,
    "Car Roof":             30,
    "Road":                 60,
    "None":                 90,
}

# =====================================
# ETHICS RISK SCORES
# =====================================

risk_scores = {
    "person":        1000,
    "dog":            800,
    "cat":            800,
    "bird":           700,
    "motorcycle":     120,
    "bicycle":        100,
    "car":             60,
    "truck":           50,
    "bus":             50,
    "bench":           20,
    "traffic light":   10,
    "stop sign":       10,
    "fire hydrant":    10,
}

OBSTACLE_CLASSES = {
    "person", "dog", "cat", "bird",
    "car", "truck", "bus", "motorcycle",
    "bicycle", "traffic light", "stop sign",
    "fire hydrant", "bench"
}

# =====================================
# SMART LANDING ZONE DETECTION (Fixed)
# =====================================

def find_best_landing_zone(image_path, obstacle_boxes):
    image = cv2.imread(image_path)
    h, w  = image.shape[:2]
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── 1. Obstacle mask ──────────────────────────────────────────
    obstacle_mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in obstacle_boxes:
        x1m = max(0, int(x1) - 20)
        y1m = max(0, int(y1) - 20)
        x2m = min(w, int(x2) + 20)
        y2m = min(h, int(y2) + 20)
        obstacle_mask[y1m:y2m, x1m:x2m] = 255

    # ── 2. Low-texture (flat) mask ────────────────────────────────
    edges      = cv2.Canny(gray, 30, 100)
    kernel     = np.ones((15, 15), np.uint8)
    edge_dense = cv2.dilate(edges, kernel, iterations=1)
    flat_mask  = cv2.bitwise_not(edge_dense)

    # ── 3. Ground-level mask (bottom 65%) ────────────────────────
    ground_mask = np.zeros((h, w), dtype=np.uint8)
    ground_mask[int(h * 0.35):, :] = 255

    # ── 4. Colour masks ───────────────────────────────────────────
    low_sat_mask = cv2.inRange(hsv,
        np.array([0,   0,  30]),
        np.array([180, 60, 220])
    )
    grass_mask = cv2.inRange(hsv,
        np.array([30, 30, 30]),
        np.array([85, 180, 180])
    )
    dirt_mask = cv2.inRange(hsv,
        np.array([5,  20, 40]),
        np.array([30, 150, 180])
    )
    colour_mask = cv2.bitwise_or(low_sat_mask, grass_mask)
    colour_mask = cv2.bitwise_or(colour_mask,  dirt_mask)

    # ── 5. Combine masks ──────────────────────────────────────────
    combined = cv2.bitwise_and(flat_mask, ground_mask)
    combined = cv2.bitwise_and(combined,  colour_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(obstacle_mask))

    # ── 6. Morphological cleanup ──────────────────────────────────
    kernel2  = np.ones((25, 25), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel2)

    # ── 7. Find contours and rank ALL zones ───────────────────────
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    ranked_zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        M  = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Classify zone type
        region = hsv[max(0, cy-20):cy+20, max(0, cx-20):cx+20]
        zone_type = "Unknown"
        if region.size > 0:
            mean_h = np.mean(region[:, :, 0])
            mean_s = np.mean(region[:, :, 1])
            if 25 < mean_h < 90 and mean_s > 30:
                zone_type = "Grass / Open Ground"
            elif mean_s < 50:
                zone_type = "Concrete / Asphalt"
            else:
                zone_type = "Dirt / Soil"

        score = SURFACE_SCORES.get(zone_type, 40)
        ranked_zones.append({
            "contour":   cnt,
            "center":    (cx, cy),
            "area":      area,
            "zone_type": zone_type,
            "score":     score,
        })

    # Sort by score descending (Feature 6)
    ranked_zones.sort(key=lambda z: z["score"], reverse=True)

    if not ranked_zones:
        return None, None, 0, "None", []

    best = ranked_zones[0]
    return best["contour"], best["center"], best["area"], best["zone_type"], ranked_zones


# =====================================
# PACKAGE SELECTION DIALOG  (Feature 7)
# =====================================

class PackageDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Package Configuration")
        self.geometry("420x520")
        self.configure(bg="#16213e")
        self.resizable(False, False)
        self.grab_set()             # Modal
        self.result = None

        # ── Header ───────────────────────────────────────────────
        tk.Label(
            self,
            text="📦  What are you carrying?",
            font=("Arial", 16, "bold"),
            bg="#16213e", fg="#e0e0ff"
        ).pack(pady=(22, 6))

        tk.Label(
            self,
            text="Select package type to calibrate landing strategy",
            font=("Arial", 10),
            bg="#16213e", fg="#8888aa"
        ).pack(pady=(0, 18))

        # ── Package buttons ───────────────────────────────────────
        self.selected = tk.StringVar(value="")
        btn_frame = tk.Frame(self, bg="#16213e")
        btn_frame.pack(fill="x", padx=30)

        for key, pkg in PACKAGE_TYPES.items():
            row = tk.Frame(btn_frame, bg="#0d1b3e", pady=10, padx=14,
                           cursor="hand2")
            row.pack(fill="x", pady=5)
            row.bind("<Button-1>", lambda e, k=key: self._select(k))

            icon = {"1": "💊", "2": "📱", "3": "📄", "4": "🍱"}[key]
            tk.Label(row, text=f"{icon}  {pkg['name']}",
                     font=("Arial", 13, "bold"),
                     bg="#0d1b3e", fg=pkg["priority_color"],
                     cursor="hand2"
            ).pack(anchor="w")
            tk.Label(row, text=pkg["description"],
                     font=("Arial", 9),
                     bg="#0d1b3e", fg="#7788aa",
                     cursor="hand2"
            ).pack(anchor="w")
            tk.Label(row, text=f"Priority: {pkg['priority']}",
                     font=("Arial", 9, "bold"),
                     bg="#0d1b3e", fg=pkg["priority_color"],
                     cursor="hand2"
            ).pack(anchor="w")

            # Highlight on hover
            def on_enter(e, r=row): r.config(bg="#1a2f5e")
            def on_leave(e, r=row): r.config(bg="#0d1b3e")
            row.bind("<Enter>", on_enter)
            row.bind("<Leave>", on_leave)
            for child in row.winfo_children():
                child.bind("<Button-1>", lambda e, k=key: self._select(k))
                child.bind("<Enter>", lambda e, r=row: r.config(bg="#1a2f5e"))
                child.bind("<Leave>", lambda e, r=row: r.config(bg="#0d1b3e"))

        # ── Confirm button ────────────────────────────────────────
        self.confirm_btn = tk.Button(
            self,
            text="  Confirm & Analyze  ",
            command=self._confirm,
            font=("Arial", 12, "bold"),
            bg="#0f3460", fg="white",
            activebackground="#1a5276",
            relief="flat", padx=16, pady=8,
            cursor="hand2", state="disabled"
        )
        self.confirm_btn.pack(pady=22)

    def _select(self, key):
        self.result = key
        self.confirm_btn.config(state="normal")
        # Visual feedback
        self.confirm_btn.config(
            text=f"  Confirm: {PACKAGE_TYPES[key]['name']}  "
        )

    def _confirm(self):
        self.destroy()


# =====================================
# MAIN ANALYSIS FUNCTION
# =====================================

def analyze_image():
    # ── Step 1: Ask for package type first (Feature 7) ───────────
    dialog = PackageDialog(root)
    root.wait_window(dialog)

    if not dialog.result:
        return  # User cancelled

    pkg_key  = dialog.result
    pkg_info = PACKAGE_TYPES[pkg_key]

    # ── Step 2: Select image ──────────────────────────────────────
    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    # ── Step 3: Run YOLO ──────────────────────────────────────────
    results          = model(file_path)
    detected_objects = []
    counts           = Counter()
    obstacle_boxes   = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name     = result.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detected_objects.append({
                "name": name,
                "x": cx, "y": cy,
                "box": (x1, y1, x2, y2)
            })
            counts[name] += 1

            if name in OBSTACLE_CLASSES:
                obstacle_boxes.append((x1, y1, x2, y2))

    # ── Step 4: Landing zone detection (with ranking fix) ─────────
    lz_contour, lz_center, lz_area, lz_type, ranked_zones = \
        find_best_landing_zone(file_path, obstacle_boxes)

    # ── Step 5: Package damage prediction (Feature 9) ─────────────
    base_damage = SURFACE_DAMAGE_RISK.get(lz_type, 40)
    damage_pct  = min(99, int(base_damage * pkg_info["damage_multiplier"]))

    if damage_pct <= 15:
        damage_label = "MINIMAL"
        damage_color = "#44cc77"
    elif damage_pct <= 40:
        damage_label = "MODERATE"
        damage_color = "#ffaa33"
    else:
        damage_label = "HIGH"
        damage_color = "#ff4444"

    # ── Step 6: Emergency / parachute decision (Feature 11) ───────
    people_count  = counts.get("person", 0)
    animal_count  = sum(counts.get(a, 0) for a in ["dog", "cat", "bird"])
    parachute_deployed = False
    parachute_reason   = ""

    # Parachute logic: deploy when no safe zone AND high-priority package
    if (lz_area < 5000 and pkg_info["priority"] in ("CRITICAL", "HIGH")):
        parachute_deployed = True
        parachute_reason   = "No safe zone + high-priority package"
    elif people_count >= 3:
        parachute_deployed = True
        parachute_reason   = "High human density detected"

    # ── Step 7: Emergency Decision Engine (Feature 10) ───────────
    human_risk   = min(100, people_count * 25)
    animal_risk  = min(100, animal_count * 20)
    pkg_risk     = damage_pct
    surface_risk = SURFACE_DAMAGE_RISK.get(lz_type, 40)
    overall_risk = int((human_risk + animal_risk + pkg_risk + surface_risk) / 4)

    # ── Step 8: Fallback target ───────────────────────────────────
    best_target = None
    lowest_risk = float("inf")
    for obj in detected_objects:
        if obj["name"] in OBSTACLE_CLASSES:
            continue
        risk = risk_scores.get(obj["name"], 5)
        if risk < lowest_risk:
            lowest_risk = risk
            best_target = obj

    # ── Step 9: Final landing decision ───────────────────────────
    if lz_center and lz_area > 5000:
        target_x = lz_center[0]
        target_y = lz_center[1]
        landing_decision = f"LAND ON {lz_type.upper()}"
        target_label  = f"LAND ({lz_type})"
        target_color  = (0, 220, 80)
    elif best_target:
        target_x = int(best_target["x"])
        target_y = int(best_target["y"])
        landing_decision = f"APPROACH {best_target['name'].upper()}"
        target_label  = f"TARGET ({best_target['name']})"
        target_color  = (0, 220, 255)
    else:
        target_x = None
        target_y = None
        landing_decision = "HOVER — NO SAFE ZONE FOUND"
        target_label  = ""
        target_color  = (200, 200, 200)

    # ── Step 10: Build annotated frame ───────────────────────────
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Draw ALL ranked zones (lighter for lower-ranked)
    for i, zone in enumerate(ranked_zones):
        alpha = 1.0 - i * 0.25
        color = (int(0 * alpha), int(180 * alpha), int(80 * alpha))
        cv2.drawContours(annotated_frame, [zone["contour"]], -1, color, 2)

    # Draw best landing zone prominently
    if lz_contour is not None:
        cv2.drawContours(annotated_frame, [lz_contour], -1, (0, 220, 80), 3)
        if lz_center:
            cv2.circle(annotated_frame, lz_center, 10, (0, 220, 80), -1)
            cv2.putText(
                annotated_frame,
                f"BEST ZONE: {lz_type} [Score:{SURFACE_SCORES.get(lz_type, 40)}]",
                (lz_center[0] + 12, lz_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2
            )

    # Draw drone
    drone_x, drone_y = 50, 50
    cv2.circle(annotated_frame, (drone_x, drone_y), 14, (255, 60, 60), -1)
    cv2.putText(
        annotated_frame, f"DRONE [{pkg_info['name']}]",
        (drone_x + 18, drone_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 2
    )

    # Draw parachute indicator
    if parachute_deployed:
        cv2.putText(
            annotated_frame, "⚠ PARACHUTE DEPLOYED",
            (10, annotated_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
        )

    # Draw path + target
    if target_x is not None:
        cv2.line(annotated_frame, (drone_x, drone_y),
                 (target_x, target_y), target_color, 3)
        cv2.circle(annotated_frame, (target_x, target_y), 20, target_color, 3)
        cv2.putText(
            annotated_frame, target_label,
            (target_x + 12, target_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, target_color, 2
        )

    # ── Step 11: Display image ────────────────────────────────────
    pil_image = Image.fromarray(annotated_frame)
    pil_image.thumbnail((1200, 800))
    photo = ImageTk.PhotoImage(pil_image)
    image_label.config(image=photo)
    image_label.image = photo

    # ── Step 12: Mission status ───────────────────────────────────
    if people_count >= 5:
        mission_status = "HIGH HUMAN DENSITY"
        status_tag     = "bad"
    elif people_count >= 2:
        mission_status = "CAUTION"
        status_tag     = "warn"
    else:
        mission_status = "SAFE"
        status_tag     = "good"

    mission_outcome = "SUCCESS" if lz_area > 5000 and damage_pct < 40 else \
                      "PARTIAL" if lz_area > 5000 else "HOVER/ABORT"

    # ── Step 13: Populate report panel ───────────────────────────
    report_text.config(state="normal")
    report_text.delete("1.0", tk.END)

    def ins(text, tag=None):
        report_text.insert(tk.END, text, tag or "normal")

    # ─ PACKAGE INFO ──────────────────────────────────────────────
    ins("PACKAGE INFORMATION\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Type         : ", "label"); ins(f"{pkg_info['name']}\n", "value")
    ins(f"  Priority     : ", "label"); ins(f"{pkg_info['priority']}\n", pkg_info["priority"].lower() if pkg_info["priority"] != "CRITICAL" else "bad")
    ins(f"  Description  : ", "label"); ins(f"{pkg_info['description'][:28]}\n", "note")
    ins(f"                 {pkg_info['description'][28:]}\n\n", "note")

    # ─ DETECTED OBJECTS ──────────────────────────────────────────
    ins("DETECTED OBJECTS\n", "header")
    ins("─" * 32 + "\n", "divider")
    if not counts:
        ins("  No objects detected.\n", "normal")
    else:
        for obj, count in counts.most_common():
            ins(f"  {obj:<22}", "label")
            ins(f"× {count}\n", "value")

    # ─ LANDING ZONE RANKING ───────────────────────────────────────
    ins("\nLANDING ZONE RANKING\n", "header")
    ins("─" * 32 + "\n", "divider")
    if ranked_zones:
        ins(f"  {'Surface':<22}{'Score':>5}  {'Area':>8}\n", "label")
        ins("  " + "·" * 30 + "\n", "divider")
        for i, zone in enumerate(ranked_zones[:4]):
            prefix = "★ " if i == 0 else f"{i+1}. "
            tag    = "good" if i == 0 else "normal"
            ins(f"  {prefix}{zone['zone_type']:<20}{zone['score']:>5}  {int(zone['area']):>7,}px\n", tag)
    else:
        ins("  No landing zones found.\n", "warn")

    # ─ ZONE ANALYSIS ─────────────────────────────────────────────
    ins("\nBEST ZONE ANALYSIS\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Zone Type    : ", "label"); ins(f"{lz_type}\n", "value")
    ins(f"  Zone Area    : ", "label")
    area_str = f"{int(lz_area):,} px²" if lz_area > 0 else "0 px² (none detected)"
    ins(f"{area_str}\n", "value" if lz_area > 0 else "warn")
    ins(f"  Zone Score   : ", "label"); ins(f"{SURFACE_SCORES.get(lz_type, 0)}/100\n", "value")
    ins(f"  Status       : ", "label")
    if lz_area > 5000:
        ins("✔ Safe Landing Zone Found\n", "good")
    else:
        ins("✘ No Significant Zone\n", "warn")

    # ─ PACKAGE DAMAGE PREDICTION ──────────────────────────────────
    ins("\nPACKAGE DAMAGE PREDICTION\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Surface Risk : ", "label"); ins(f"{base_damage}%\n", "value")
    ins(f"  Pkg Factor   : ", "label"); ins(f"×{pkg_info['damage_multiplier']}\n", "value")
    ins(f"  Damage Risk  : ", "label"); ins(f"{damage_pct}% — {damage_label}\n",
        "good" if damage_pct <= 15 else "warn" if damage_pct <= 40 else "bad")

    # ─ EMERGENCY RISK ASSESSMENT ─────────────────────────────────
    ins("\nEMERGENCY RISK ASSESSMENT\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Human Risk   : ", "label"); ins(f"{human_risk}%\n", "value")
    ins(f"  Animal Risk  : ", "label"); ins(f"{animal_risk}%\n", "value")
    ins(f"  Package Risk : ", "label"); ins(f"{pkg_risk}%\n", "value")
    ins(f"  Surface Risk : ", "label"); ins(f"{surface_risk}%\n", "value")
    ins(f"  Overall Risk : ", "label"); ins(
        f"{overall_risk}%\n",
        "good" if overall_risk < 30 else "warn" if overall_risk < 60 else "bad"
    )

    # ─ PARACHUTE SYSTEM ──────────────────────────────────────────
    ins("\nPARACHUTE SYSTEM\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Status       : ", "label")
    if parachute_deployed:
        ins("🪂 DEPLOYED\n", "bad")
        ins(f"  Reason       : ", "label"); ins(f"{parachute_reason}\n", "warn")
        ins(f"  Effect       : ", "label"); ins("Impact reduced, pkg protected\n", "good")
    else:
        ins("✔ Not Required\n", "good")

    # ─ DECISION ──────────────────────────────────────────────────
    ins("\nDECISION\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Action       : ", "label"); ins(f"{landing_decision}\n", "decision")
    ins(f"  Mission      : ", "label"); ins(f"{mission_status}\n", status_tag)

    # ─ MISSION OUTCOME REPORT ────────────────────────────────────
    ins("\nMISSION OUTCOME REPORT\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(f"  Humans       : ", "label"); ins(f"{people_count} detected\n", "value")
    ins(f"  Animals      : ", "label"); ins(f"{animal_count} detected\n", "value")
    ins(f"  Package Type : ", "label"); ins(f"{pkg_info['name']}\n", "value")
    ins(f"  Landing Zone : ", "label"); ins(f"{lz_type}\n", "value")
    ins(f"  Damage Risk  : ", "label"); ins(f"{damage_pct}%\n", "value")
    ins(f"  Parachute    : ", "label"); ins(f"{'YES' if parachute_deployed else 'NO'}\n", "value")
    ins(f"  Outcome      : ", "label"); ins(
        f"{mission_outcome}\n",
        "good" if mission_outcome == "SUCCESS" else
        "warn" if mission_outcome == "PARTIAL" else "bad"
    )

    ins("\nNOTE\n", "header")
    ins("─" * 32 + "\n", "divider")
    ins(
        "  Zones ranked by texture,\n"
        "  colour, position & obstacle\n"
        "  proximity. Package type\n"
        "  adjusts damage thresholds.\n",
        "note"
    )

    report_text.config(state="disabled")
    # Scroll to top so report is fully visible
    report_text.yview_moveto(0)


# =====================================
# GUI SETUP
# =====================================

root = tk.Tk()
root.title("Drone Ethics Analyzer — Research Prototype")
root.state("zoomed")
root.configure(bg="#1a1a2e")

# ── Title bar ─────────────────────────────────────────────────────
title_frame = tk.Frame(root, bg="#16213e", pady=8)
title_frame.pack(fill="x")

tk.Label(
    title_frame,
    text="Drone Ethics Analyzer",
    font=("Arial", 24, "bold"),
    bg="#16213e", fg="#e0e0ff"
).pack()

tk.Label(
    title_frame,
    text="Smart Landing Zone Detection  |  Package-Aware  |  Ethical Obstacle Avoidance",
    font=("Arial", 11),
    bg="#16213e", fg="#8888aa"
).pack()

# ── Select button ─────────────────────────────────────────────────
btn_frame = tk.Frame(root, bg="#1a1a2e", pady=8)
btn_frame.pack()

tk.Button(
    btn_frame,
    text="  📦  Select Package & Analyze Image  ",
    command=analyze_image,
    font=("Arial", 13, "bold"),
    bg="#0f3460", fg="white",
    activebackground="#1a5276",
    relief="flat", padx=16, pady=6, cursor="hand2"
).pack()

# ── Main layout ───────────────────────────────────────────────────
main_frame = tk.Frame(root, bg="#1a1a2e")
main_frame.pack(fill="both", expand=True, padx=12, pady=8)

# Left — image
left_frame = tk.Frame(main_frame, bg="#0d0d1a", bd=2, relief="sunken")
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

image_label = tk.Label(left_frame, bg="#0d0d1a")
image_label.pack(expand=True)

# Right — scrollable report (wider + starts at top)
right_frame = tk.Frame(main_frame, bg="#1a1a2e", width=400)
right_frame.pack(side="right", fill="y")
right_frame.pack_propagate(False)

tk.Label(
    right_frame,
    text="Analysis Report",
    font=("Arial", 14, "bold"),
    bg="#1a1a2e", fg="#e0e0ff"
).pack(pady=(0, 4))

text_frame = tk.Frame(right_frame, bg="#1a1a2e")
text_frame.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(text_frame)
scrollbar.pack(side="right", fill="y")

report_text = tk.Text(
    text_frame,
    width=44,
    font=("Consolas", 11),
    wrap="word",
    bg="#0d0d1a",
    fg="#c8c8e8",
    insertbackground="white",
    selectbackground="#2244aa",
    relief="flat",
    padx=8, pady=6,
    yscrollcommand=scrollbar.set,
    state="disabled"
)
report_text.pack(side="left", fill="both", expand=True)
scrollbar.config(command=report_text.yview)

# ── Text tags ─────────────────────────────────────────────────────
report_text.tag_configure("header",   foreground="#7ec8e3", font=("Consolas", 11, "bold"))
report_text.tag_configure("divider",  foreground="#334466")
report_text.tag_configure("label",    foreground="#8899bb")
report_text.tag_configure("value",    foreground="#ddeeff")
report_text.tag_configure("good",     foreground="#44cc77", font=("Consolas", 11, "bold"))
report_text.tag_configure("warn",     foreground="#ffaa33", font=("Consolas", 11, "bold"))
report_text.tag_configure("bad",      foreground="#ff4444", font=("Consolas", 11, "bold"))
report_text.tag_configure("decision", foreground="#ffdd66", font=("Consolas", 11, "bold"))
report_text.tag_configure("normal",   foreground="#b0b8cc")
report_text.tag_configure("note",     foreground="#7a7aaa", font=("Consolas", 10))
report_text.tag_configure("medium",   foreground="#ffdd00", font=("Consolas", 11, "bold"))
report_text.tag_configure("high",     foreground="#ff8800", font=("Consolas", 11, "bold"))
report_text.tag_configure("low",      foreground="#44cc77", font=("Consolas", 11, "bold"))
report_text.tag_configure("critical", foreground="#ff4444", font=("Consolas", 11, "bold"))

root.mainloop()

