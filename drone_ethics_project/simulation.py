import pygame
import pandas as pd
import pickle
import os
import sys
import math

# =====================================
# CONFIGURATION & CONSTANTS
# =====================================

WIDTH,  HEIGHT  = 1100, 650
FPS             = 60
DRONE_SPEED     = 2.5

# ── Colours ────────────────────────────────────────────────────────
C_BG         = (15,  20,  40)
C_GROUND     = (30,  35,  55)
C_GRASS      = (34,  139, 34)
C_SOIL       = (139, 90,  43)
C_ROAD       = (80,  80,  90)
C_HUMAN      = (220, 60,  60)
C_CAR        = (100, 100, 120)
C_ANIMAL     = (60,  180, 60)
C_DRONE      = (80,  140, 255)
C_PATH       = (255, 220, 50)
C_WHITE      = (255, 255, 255)
C_PANEL      = (20,  25,  50)
C_ACCENT     = (80,  140, 255)
C_GOOD       = (50,  200, 100)
C_WARN       = (255, 170, 30)
C_BAD        = (220, 60,  60)
C_PARA       = (255, 140, 0)
C_LABEL      = (150, 160, 200)

# ── Package definitions ────────────────────────────────────────────
PACKAGES = {
    "1": {"name": "Medicine",     "priority": "Critical", "priority_val": 4, "colour": (220, 60,  60)},
    "2": {"name": "Electronics",  "priority": "Medium",   "priority_val": 2, "colour": (80,  180, 220)},
    "3": {"name": "Documents",    "priority": "High",     "priority_val": 3, "colour": (220, 180, 50)},
    "4": {"name": "Food",         "priority": "Low",      "priority_val": 1, "colour": (80,  200, 80)},
}

# ── Landing zone definitions (priority order) ──────────────────────
LANDING_ZONES = [
    {"name": "Grass",       "score": 100, "damage": 5,  "colour": C_GRASS,  "key": "grass"},
    {"name": "Soil / Dirt", "score": 90,  "damage": 10, "colour": C_SOIL,   "key": "soil"},
    {"name": "Empty Road",  "score": 70,  "damage": 20, "colour": C_ROAD,   "key": "road"},
    {"name": "Car Roof",    "score": 40,  "damage": 35, "colour": C_CAR,    "key": "car"},
    {"name": "Tree / Bush", "score": 30,  "damage": 50, "colour": C_ANIMAL, "key": "tree"},
]

# =====================================
# LOAD ML MODEL
# =====================================

def load_model():
    path = os.path.join(os.path.dirname(__file__), "drone_model.pkl")
    if not os.path.exists(path):
        print("[ERROR] drone_model.pkl not found. Run train_model.py first.")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)

# =====================================
# ETHICAL DECISION ENGINE  (Feature 10)
# Evaluates all risks and picks the
# best overall landing strategy
# =====================================

def ethical_decision_engine(pedestrians, animals, package_priority,
                              best_zone, failure_mode):
    """
    Returns (decision_text, target_zone_key, use_parachute)
    """
    use_parachute = failure_mode != "none"

    # Critical failures always trigger parachute
    if failure_mode in ("battery", "motor", "comms"):
        use_parachute = True

    # Medicine with high human density → maximise caution
    if pedestrians >= 3 and package_priority == 4:
        return "EMERGENCY LAND — MEDICINE CRITICAL", best_zone, use_parachute

    if pedestrians >= 3:
        return "HIGH HUMAN RISK — SEEK SAFE ZONE", best_zone, use_parachute

    if animals >= 3:
        return "ANIMAL RISK — REDIRECT TO SAFE ZONE", best_zone, use_parachute

    if best_zone["score"] >= 90:
        return "SAFE LANDING — OPTIMAL ZONE", best_zone, use_parachute

    if best_zone["score"] >= 70:
        return "ACCEPTABLE LANDING — ROAD ZONE", best_zone, use_parachute

    return "RISKY LANDING — NO IDEAL ZONE", best_zone, use_parachute

# =====================================
# DAMAGE PREDICTION  (Feature 9)
# =====================================

def predict_damage(zone_damage_base, package_priority, use_parachute):
    """
    Adjusts base surface damage by package fragility and parachute use.
    """
    fragility_mult = {4: 1.4, 3: 1.1, 2: 1.2, 1: 0.8}  # medicine most fragile
    damage = zone_damage_base * fragility_mult.get(package_priority, 1.0)
    if use_parachute:
        damage *= 0.4          # parachute cuts impact significantly
    return min(int(damage), 100)

# =====================================
# DRAW HELPERS
# =====================================

def draw_text(surface, text, x, y, font, colour=C_WHITE, align="left"):
    surf = font.render(text, True, colour)
    rect = surf.get_rect()
    if align == "center":
        rect.centerx = x
        rect.y = y
    elif align == "right":
        rect.right = x
        rect.y = y
    else:
        rect.x = x
        rect.y = y
    surface.blit(surf, rect)

def draw_panel(surface, x, y, w, h, colour=C_PANEL, alpha=220, radius=10):
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (*colour, alpha), (0, 0, w, h), border_radius=radius)
    surface.blit(panel, (x, y))

def draw_drone(surface, x, y, size=22, parachute=False):
    # Body
    pygame.draw.circle(surface, C_DRONE, (int(x), int(y)), size)
    pygame.draw.circle(surface, C_WHITE, (int(x), int(y)), size, 2)
    # Arms
    for angle in [45, 135, 225, 315]:
        rad  = math.radians(angle)
        ex   = int(x + math.cos(rad) * (size + 10))
        ey   = int(y + math.sin(rad) * (size + 10))
        pygame.draw.line(surface, C_LABEL, (int(x), int(y)), (ex, ey), 3)
        pygame.draw.circle(surface, C_WARN, (ex, ey), 5)
    # Parachute
    if parachute:
        pts = [
            (int(x),          int(y) - size - 10),
            (int(x) - 35,     int(y) - size - 55),
            (int(x) + 35,     int(y) - size - 55),
        ]
        pygame.draw.polygon(surface, C_PARA, pts)
        pygame.draw.polygon(surface, C_WHITE, pts, 2)
        for px, py in [(int(x) - 35, int(y) - size - 55),
                       (int(x) + 35, int(y) - size - 55)]:
            pygame.draw.line(surface, C_PARA, (int(x), int(y) - size), (px, py), 2)

def draw_human(surface, x, y):
    # Body
    pygame.draw.rect(surface, C_HUMAN, (x - 14, y - 10, 28, 34))
    # Head
    pygame.draw.circle(surface, C_HUMAN, (x, y - 20), 13)
    pygame.draw.circle(surface, C_WHITE, (x, y - 20), 13, 2)

def draw_car(surface, x, y):
    pygame.draw.rect(surface, C_CAR, (x - 38, y - 15, 76, 30), border_radius=5)
    pygame.draw.rect(surface, (60, 60, 80), (x - 24, y - 28, 48, 18), border_radius=4)
    for wx, wy in [(x - 26, y + 12), (x + 26, y + 12)]:
        pygame.draw.circle(surface, (30, 30, 30), (wx, wy), 9)
        pygame.draw.circle(surface, (180, 180, 180), (wx, wy), 5)

def draw_animal(surface, x, y):
    # Simple dog-like shape
    pygame.draw.ellipse(surface, C_ANIMAL, (x - 24, y - 10, 48, 22))
    pygame.draw.circle(surface, C_ANIMAL, (x + 24, y - 14), 12)
    pygame.draw.circle(surface, C_WHITE,  (x + 24, y - 14), 12, 2)

def draw_zone(surface, zone_info, x, y, w, h, best=False):
    colour = zone_info["colour"]
    pygame.draw.rect(surface, colour, (x, y, w, h), border_radius=6)
    if best:
        pygame.draw.rect(surface, C_WHITE, (x, y, w, h), 3, border_radius=6)

def progress_bar(surface, x, y, w, h, value, max_val, colour):
    pygame.draw.rect(surface, (40, 40, 60), (x, y, w, h), border_radius=4)
    fill = int((value / max_val) * w)
    if fill > 0:
        pygame.draw.rect(surface, colour, (x, y, fill, h), border_radius=4)
    pygame.draw.rect(surface, C_LABEL, (x, y, w, h), 1, border_radius=4)

# =====================================
# INPUT SCREEN
# =====================================

def input_screen(screen, clock, font_big, font_med, font_sm):

    fields = [
        {"label": "Number of Pedestrians",  "value": "", "key": "ped"},
        {"label": "Number of Cars",          "value": "", "key": "car"},
        {"label": "Number of Animals",       "value": "", "key": "ani"},
    ]
    active  = 0
    package = "1"
    failure = "none"

    failure_opts = [
        ("none",    "None (Normal Flight)"),
        ("battery", "Battery Failure"),
        ("motor",   "Motor Failure"),
        ("comms",   "Communication Failure"),
    ]
    fail_idx = 0

    while True:
        clock.tick(FPS)
        screen.fill(C_BG)

        draw_text(screen, "DRONE ETHICS SIMULATOR", WIDTH // 2, 30,
                  font_big, C_ACCENT, "center")
        draw_text(screen, "Ethical Emergency Landing Framework", WIDTH // 2, 72,
                  font_sm, C_LABEL, "center")

        # ── Numeric inputs ─────────────────────────────────────────
        for i, field in enumerate(fields):
            fy = 130 + i * 70
            colour = C_ACCENT if i == active else C_LABEL
            draw_panel(screen, WIDTH // 2 - 200, fy, 400, 48, alpha=180)
            draw_text(screen, field["label"], WIDTH // 2 - 190, fy + 6,
                      font_sm, C_LABEL)
            val_text = field["value"] + ("|" if i == active else "")
            draw_text(screen, val_text, WIDTH // 2 + 100, fy + 10,
                      font_med, colour)

        # ── Package selection ──────────────────────────────────────
        draw_text(screen, "Package Type:", WIDTH // 2 - 200, 360, font_sm, C_LABEL)
        for k, pkg in PACKAGES.items():
            px = WIDTH // 2 - 200 + (int(k) - 1) * 120
            sel = k == package
            draw_panel(screen, px, 385, 110, 42,
                       colour=pkg["colour"] if sel else C_GROUND, alpha=200)
            draw_text(screen, pkg["name"], px + 55, 397,
                      font_sm, C_WHITE if sel else C_LABEL, "center")

        # ── Failure mode ───────────────────────────────────────────
        draw_text(screen, "Failure Mode:", WIDTH // 2 - 200, 450, font_sm, C_LABEL)
        fi, fdesc = failure_opts[fail_idx]
        draw_panel(screen, WIDTH // 2 - 200, 475, 400, 42, alpha=180)
        draw_text(screen, f"< {fdesc} >", WIDTH // 2, 488,
                  font_sm, C_WARN if fi != "none" else C_GOOD, "center")

        # ── Start button ───────────────────────────────────────────
        draw_panel(screen, WIDTH // 2 - 120, 545, 240, 50,
                   colour=(20, 80, 160), alpha=230, radius=12)
        draw_text(screen, "LAUNCH SIMULATION", WIDTH // 2, 558,
                  font_sm, C_WHITE, "center")

        draw_text(screen, "TAB = next field   |   ENTER = launch   |   ESC = quit",
                  WIDTH // 2, 615, font_sm, C_LABEL, "center")

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                elif event.key == pygame.K_TAB:
                    active = (active + 1) % len(fields)

                elif event.key == pygame.K_RETURN:
                    # Validate
                    try:
                        vals = [int(f["value"]) for f in fields]
                        return vals[0], vals[1], vals[2], package, failure_opts[fail_idx][0]
                    except ValueError:
                        for f in fields:
                            if not f["value"]:
                                f["value"] = "0"

                elif event.key == pygame.K_BACKSPACE:
                    fields[active]["value"] = fields[active]["value"][:-1]

                elif event.unicode.isdigit():
                    if len(fields[active]["value"]) < 2:
                        fields[active]["value"] += event.unicode

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                # Package buttons
                for k in PACKAGES:
                    px = WIDTH // 2 - 200 + (int(k) - 1) * 120
                    if px <= mx <= px + 110 and 385 <= my <= 427:
                        package = k

                # Failure cycle
                if WIDTH // 2 - 200 <= mx <= WIDTH // 2 + 200 and 475 <= my <= 517:
                    fail_idx = (fail_idx + 1) % len(failure_opts)
                    failure  = failure_opts[fail_idx][0]

                # Launch button
                if WIDTH // 2 - 120 <= mx <= WIDTH // 2 + 120 and 545 <= my <= 595:
                    try:
                        vals = [int(f["value"] or 0) for f in fields]
                        return vals[0], vals[1], vals[2], package, failure_opts[fail_idx][0]
                    except ValueError:
                        pass

# =====================================
# MAIN SIMULATION
# =====================================

def run_simulation(pedestrians, cars, animals, package_key, failure_mode, ml_model):

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Ethics Simulator — Research Prototype")
    clock  = pygame.time.Clock()

    font_big = pygame.font.SysFont("Arial", 28, bold=True)
    font_med = pygame.font.SysFont("Arial", 22, bold=True)
    font_sm  = pygame.font.SysFont("Arial", 18)
    font_xs  = pygame.font.SysFont("Arial", 15)

    # ── Input screen ───────────────────────────────────────────────
    pedestrians, cars, animals, package_key, failure_mode = \
        input_screen(screen, clock, font_big, font_med, font_sm)

    pkg = PACKAGES[package_key]

    # ── ML prediction ──────────────────────────────────────────────
    pkg_priority = pkg["priority_val"]
    # Use best possible landing zone score for ML feature
    test_df = pd.DataFrame(
        [[pedestrians, cars, animals, pkg_priority, 100]],
        columns=["pedestrians", "cars", "animals", "package_priority", "landing_score"]
    )
    ml_decision = ml_model.predict(test_df)[0]

    # ── Landing zone ranking  (Feature 6) ─────────────────────────
    # Remove zones blocked by detected objects
    available_zones = []
    for z in LANDING_ZONES:
        if z["key"] == "car"  and cars == 0:   continue
        available_zones.append(z)
    if not available_zones:
        available_zones = LANDING_ZONES[:]

    best_zone = available_zones[0]   # highest score first

    # ── Ethical decision engine  (Feature 10) ─────────────────────
    decision_text, chosen_zone, use_parachute = ethical_decision_engine(
        pedestrians, animals, pkg_priority, best_zone, failure_mode
    )

    # ── Damage prediction  (Feature 9) ────────────────────────────
    damage_pct = predict_damage(chosen_zone["damage"], pkg_priority, use_parachute)

    # ── Mission outcome  (Feature 12) ─────────────────────────────
    if damage_pct <= 15:
        outcome = "SUCCESS"
        outcome_c = C_GOOD
    elif damage_pct <= 40:
        outcome = "PARTIAL SUCCESS"
        outcome_c = C_WARN
    else:
        outcome = "PACKAGE AT RISK"
        outcome_c = C_BAD

    # ── Scene layout ───────────────────────────────────────────────
    SIM_W   = 680           # simulation viewport width
    PANEL_X = SIM_W + 10    # right panel x

    # Object positions (in simulation area)
    human_pos  = (200, 300)
    car_pos    = (380, 310)
    animal_pos = (530, 300)

    # Landing zone positions on ground
    lz_rects = {
        "grass": pygame.Rect(60,  370, 120, 60),
        "soil":  pygame.Rect(200, 375, 100, 55),
        "road":  pygame.Rect(320, 378, 110, 50),
        "car":   pygame.Rect(450, 382,  90, 46),
        "tree":  pygame.Rect(560, 380,  80, 48),
    }

    # Drone start
    drone_x, drone_y = 60.0, 180.0

    # Target = centre of best landing zone
    lz_r        = lz_rects.get(chosen_zone["key"], lz_rects["road"])
    target_x    = float(lz_r.centerx)
    target_y    = float(lz_r.centery)

    reached     = False
    landed      = False
    land_timer  = 0
    show_report = False

    running = True
    while running:
        clock.tick(FPS)

        # ── Events ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    run_simulation(0, 0, 0, "1", "none", ml_model)
                    return

        # ── Move drone ────────────────────────────────────────────
        if not reached:
            dx = target_x - drone_x
            dy = target_y - drone_y
            dist = math.hypot(dx, dy)
            if dist < DRONE_SPEED + 1:
                drone_x, drone_y = target_x, target_y
                reached = True
            else:
                drone_x += (dx / dist) * DRONE_SPEED
                drone_y += (dy / dist) * DRONE_SPEED
        else:
            if not landed:
                land_timer += 1
                if land_timer > FPS * 1.5:
                    landed      = True
                    show_report = True

        # ── Draw background ───────────────────────────────────────
        screen.fill(C_BG)

        # Sky gradient (simulation area)
        for i in range(350):
            c = (max(15, 15 + i // 10), max(20, 20 + i // 8), max(40, 40 + i // 6))
            pygame.draw.line(screen, c, (0, i), (SIM_W, i))

        # Ground strip
        pygame.draw.rect(screen, C_GROUND, (0, 350, SIM_W, HEIGHT - 350))

        # ── Draw landing zones (ranked) ────────────────────────────
        for z in LANDING_ZONES:
            r = lz_rects.get(z["key"])
            if r:
                is_best = z["key"] == chosen_zone["key"]
                draw_zone(screen, z, r.x, r.y, r.w, r.h, best=is_best)
                draw_text(screen, z["name"],  r.centerx, r.y + 4,  font_xs, C_WHITE, "center")
                draw_text(screen, f"{z['score']}pts", r.centerx, r.y + 20, font_xs, C_WARN, "center")
                draw_text(screen, f"{z['damage']}%dmg", r.centerx, r.y + 36, font_xs, C_BAD, "center")

        # ── Draw scene objects ─────────────────────────────────────
        for _ in range(pedestrians):
            draw_human(screen, human_pos[0] + _ * 30, human_pos[1])

        for _ in range(min(cars, 3)):
            draw_car(screen, car_pos[0] + _ * 85, car_pos[1])

        for _ in range(min(animals, 3)):
            draw_animal(screen, animal_pos[0] + _ * 55, animal_pos[1])

        # Labels
        if pedestrians > 0:
            draw_text(screen, f"Humans ({pedestrians})", human_pos[0], human_pos[1] + 30, font_xs, C_HUMAN)
        if cars > 0:
            draw_text(screen, f"Cars ({cars})", car_pos[0], car_pos[1] + 35, font_xs, C_LABEL)
        if animals > 0:
            draw_text(screen, f"Animals ({animals})", animal_pos[0], animal_pos[1] + 30, font_xs, C_ANIMAL)

        # ── Draw flight path ───────────────────────────────────────
        pygame.draw.line(screen, (*C_PATH, 120),
                         (int(drone_x), int(drone_y)),
                         (int(target_x), int(target_y)), 2)

        # Target marker
        pygame.draw.circle(screen, C_GOOD, (int(target_x), int(target_y)), 14, 3)

        # ── Draw drone ────────────────────────────────────────────
        draw_drone(screen, drone_x, drone_y,
                   parachute=(use_parachute and not reached))

        # Package indicator on drone
        if not reached:
            pygame.draw.rect(screen,
                             pkg["colour"],
                             (int(drone_x) - 8, int(drone_y) + 24, 16, 12),
                             border_radius=3)

        # ── Separator ─────────────────────────────────────────────
        pygame.draw.line(screen, C_ACCENT, (SIM_W, 0), (SIM_W, HEIGHT), 2)

        # ── RIGHT PANEL ───────────────────────────────────────────
        PX = PANEL_X + 5
        PW = WIDTH - PANEL_X - 10
        py = 10

        # Title
        draw_panel(screen, PANEL_X, py, PW, 38, colour=(10, 20, 60), alpha=220)
        draw_text(screen, "MISSION REPORT", PX + PW // 2, py + 9,
                  font_med, C_ACCENT, "center")
        py += 46

        # ── Scenario ──────────────────────────────────────────────
        draw_panel(screen, PANEL_X, py, PW, 80, alpha=160)
        draw_text(screen, "SCENARIO", PX + 5, py + 4, font_xs, C_LABEL)
        draw_text(screen, f"Humans   : {pedestrians}", PX + 5, py + 20, font_xs, C_WHITE)
        draw_text(screen, f"Cars     : {cars}",        PX + 5, py + 36, font_xs, C_WHITE)
        draw_text(screen, f"Animals  : {animals}",     PX + 5, py + 52, font_xs, C_WHITE)
        py += 88

        # ── Package  (Feature 7 & 8) ──────────────────────────────
        draw_panel(screen, PANEL_X, py, PW, 60, alpha=160)
        draw_text(screen, "PACKAGE", PX + 5, py + 4, font_xs, C_LABEL)
        draw_text(screen, f"Type     : {pkg['name']}",     PX + 5, py + 20, font_xs, C_WHITE)
        pri_c = C_BAD if pkg["priority"] == "Critical" else \
                C_WARN if pkg["priority"] in ("High", "Medium") else C_GOOD
        draw_text(screen, f"Priority : {pkg['priority']}", PX + 5, py + 38, font_xs, pri_c)
        py += 68

        # ── Landing zone ranking  (Feature 6) ─────────────────────
        draw_panel(screen, PANEL_X, py, PW, 130, alpha=160)
        draw_text(screen, "LANDING ZONE RANKING", PX + 5, py + 4, font_xs, C_LABEL)
        for i, z in enumerate(LANDING_ZONES[:5]):
            zy  = py + 22 + i * 21
            sel = z["key"] == chosen_zone["key"]
            clr = C_GOOD if sel else C_WHITE
            progress_bar(screen, PX + 5, zy + 4, 60, 12, z["score"], 100, z["colour"])
            draw_text(screen, z["name"],       PX + 72,  zy, font_xs, clr)
            draw_text(screen, f"{z['score']}", PX + 172, zy, font_xs, clr)
            if sel:
                draw_text(screen, "◀", PX + 190, zy, font_xs, C_GOOD)
        py += 138

        # ── Failure mode  (Feature 11 — parachute) ────────────────
        draw_panel(screen, PANEL_X, py, PW, 44, alpha=160)
        draw_text(screen, "FAILURE MODE", PX + 5, py + 4, font_xs, C_LABEL)
        fm_text = failure_mode.upper() if failure_mode != "none" else "None"
        fm_c    = C_BAD if failure_mode != "none" else C_GOOD
        draw_text(screen, fm_text, PX + 5, py + 22, font_xs, fm_c)
        if use_parachute:
            draw_text(screen, "PARACHUTE: DEPLOYED", PX + 90, py + 22, font_xs, C_PARA)
        py += 52

        # ── ML Decision ───────────────────────────────────────────
        draw_panel(screen, PANEL_X, py, PW, 44, alpha=160)
        draw_text(screen, "ML MODEL DECISION", PX + 5, py + 4, font_xs, C_LABEL)
        draw_text(screen, ml_decision.replace("_", " ").upper(), PX + 5, py + 22, font_xs, C_ACCENT)
        py += 52

        # ── Ethical decision  ──────────────────────────────────────
        draw_panel(screen, PANEL_X, py, PW, 44, alpha=160)
        draw_text(screen, "ETHICAL ENGINE DECISION", PX + 5, py + 4, font_xs, C_LABEL)
        draw_text(screen, decision_text[:30], PX + 5, py + 22, font_xs, C_WARN)
        py += 52

        # ── Damage prediction  (Feature 9) ────────────────────────
        draw_panel(screen, PANEL_X, py, PW, 50, alpha=160)
        draw_text(screen, "PACKAGE DAMAGE PREDICTION", PX + 5, py + 4, font_xs, C_LABEL)
        dmg_c = C_GOOD if damage_pct <= 15 else C_WARN if damage_pct <= 40 else C_BAD
        progress_bar(screen, PX + 5, py + 24, PW - 15, 14, damage_pct, 100, dmg_c)
        draw_text(screen, f"{damage_pct}%", PX + PW // 2, py + 22, font_xs, dmg_c, "center")
        py += 58

        # ── Mission outcome  (Feature 12) ─────────────────────────
        if show_report:
            draw_panel(screen, PANEL_X, py, PW, 80,
                       colour=(10, 40, 10) if outcome == "SUCCESS" else (40, 20, 10),
                       alpha=210)
            draw_text(screen, "MISSION OUTCOME", PX + 5, py + 4, font_xs, C_LABEL)
            draw_text(screen, outcome,           PX + PW // 2, py + 26,
                      font_med, outcome_c, "center")
            lz_label = f"Landed : {chosen_zone['name']}"
            para_lbl = "Parachute : YES" if use_parachute else "Parachute : NO"
            draw_text(screen, lz_label,  PX + 5, py + 52, font_xs, C_WHITE)
            draw_text(screen, para_lbl,  PX + 5, py + 66, font_xs,
                      C_PARA if use_parachute else C_LABEL)
            py += 88

            draw_text(screen, "Press R to restart  |  ESC to quit",
                      PANEL_X + PW // 2, py + 8, font_xs, C_LABEL, "center")

        # ── Status bar at bottom of sim area ──────────────────────
        status = "FLYING TO LANDING ZONE..." if not reached else \
                 ("LANDING..." if not landed else "LANDED SAFELY")
        pygame.draw.rect(screen, (10, 15, 35), (0, HEIGHT - 30, SIM_W, 30))
        draw_text(screen, status, SIM_W // 2, HEIGHT - 24,
                  font_xs, C_GOOD if landed else C_WARN, "center")

        pygame.display.flip()

    pygame.quit()


# =====================================
# ENTRY POINT
# =====================================

if __name__ == "__main__":
    ml_model = load_model()
    run_simulation(0, 0, 0, "1", "none", ml_model)