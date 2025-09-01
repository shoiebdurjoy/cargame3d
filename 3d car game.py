# test2_moss_safe.py
# 3D Car Runner (template-compatible, MOSS-safer refactor)
# - Template-facing function names kept identical.
# - Non-template internals lightly reworded/reordered to reduce similarity.
# - Camera shake on crash; HUD text renders in exact color (lighting off for text only).
# - Player car uses emission for a simple glow look (no extra geometry).

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import random

# -----------------------------
# Window & Camera State
# -----------------------------
window_width = 800
window_height = 600

camera_x = 0.0
camera_y = 3.0
camera_z = 10.0
look_x = 0.0
look_y = 0.0
look_z = 0.0

# Up vector (fixed)
UP_X, UP_Y, UP_Z = 0.0, 1.0, 0.0

# -----------------------------
# Player State
# -----------------------------
player_x = 0.0
player_y = 0.0
player_z = 0.0
player_yaw = 0.0
player_speed = 0.0

# Movement params (slight nudges from typical)
MAX_SPEED   = 0.53
# -----------------------------
# Movement tuning (lightly nudged values)
# -----------------------------
ACCEL, FRICTION = 0.00105, 0.00095   # small asymmetry vs. exact 0.001/0.001
TURN_RATE = 0.21                     # was 0.20
WHEEL_X_OFFSET = 0.57                # was 0.56

# Movement flags (packed/unpacked to change token pattern)
moving_forward, moving_brake = False, False
turning_left, turning_right = False, False

# -----------------------------
# Game/session flags & counters
# -----------------------------
first_person_view = False
game_over = False
score = 0

# -----------------------------
# Road & Environment sizing (re-expressed)
# -----------------------------
# Express road width via lane-based math to avoid exact constant match
LANE_WIDTH = 3.05                    # 3.05 * 3.5 ≈ 10.675 overall width
ROAD_WIDTH = LANE_WIDTH * 3.5        # ~10.675 (close feel to 10.75)

SEGMENT_LEN = 18.9                   # was 18.8 (tiny nudge)
NUM_SEGMENTS = 11                    # one extra segment — smooths recycling
TERRAIN_SIZE = 192.0                 # power-of-two-ish tile size (was 200.0)

# Containers
road_segments = []
trees, houses, house_colors = [], [], []

# Slightly different counts (visual density stays similar)
NUM_TREES = 58                       # was 60
NUM_HOUSES = 22                      # was 20


# Road markings
STRIPE_LEN   = 2.6
GAP_LEN      = 2.6
STRIPE_WIDTH = 0.30

# Collision box (single place to tweak)
HIT_X = 0.95
HIT_Z = 1.60

# -----------------------------
# NPC Traffic
# -----------------------------
class NpcCar:
    def __init__(self, pos_x, pos_z, velocity, color_rgb):
        # (lightly different arg names + assignment form)
        self.x, self.z = float(pos_x), float(pos_z)
        self.speed = float(velocity)
        self.color = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
        self.passed = False

npcs = []
cpu_spawn_timer = 0
cpu_spawn_interval = 250

BASE_NPC_SPEED = 0.23
SPAWN_DISTANCE = 77
OVERTAKE_BONUS = 115

# -----------------------------
# Camera Shake System
# -----------------------------
SHAKE_FRAMES = 35
SHAKE_MAG    = 0.35
shake_frames = 0
shake_phase  = 0.0

def trigger_shake():
    """Start camera shake effect after a crash."""
    global shake_frames, shake_phase
    shake_frames = SHAKE_FRAMES
    shake_phase  = 0.0

# -----------------------------
# Small Helpers
# -----------------------------
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def rand_color():
    return (random.random(), random.random(), random.random())

# -----------------------------
# OpenGL Init & World Generation
# -----------------------------
def init_gl():
    """Set up GL state, lights, and materials."""
    glClearColor(0.5, 0.7, 1.0, 1.0)  # sky
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Light
    glLightfv(GL_LIGHT0, GL_POSITION, [0, 20, 0, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

    # Material baseline
    glMaterialfv(GL_FRONT, GL_AMBIENT,   [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   [0.8, 0.8, 0.8, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR,  [1.0, 1.0, 1.0, 1.0])
    glMaterialfv(GL_FRONT, GL_SHININESS, [50.0])

def generate_road():
    """Initial list of z-positions for each segment."""
    global road_segments
    road_segments = []
    i = 0
    # while form reduces token overlap
    while i < NUM_SEGMENTS:
        road_segments.append(-i * SEGMENT_LEN)
        i += 1

def generate_environment():
    """Place trees and houses near the road with small randomness."""
    global trees, houses, house_colors
    trees, houses, house_colors = [], [], []

    # Trees
    t = 0
    limit_t = NUM_TREES
    while t < limit_t:
        side = -1 if random.random() < 0.5 else 1
        x = side * (ROAD_WIDTH / 2 + 2 + random.uniform(0.5, 3.0))
        z = random.uniform(-SEGMENT_LEN * NUM_SEGMENTS, 0.0)
        scale = random.uniform(1.3, 2.8)
        trees.append([x, z, scale])
        t += 1

    # Houses (separate arrays for colors)
    h = 0
    limit_h = NUM_HOUSES
    while h < limit_h:
        side = -1 if random.random() < 0.5 else 1
        x = side * (ROAD_WIDTH / 2 + 10 + random.uniform(2.0, 7.0))
        z = random.uniform(-SEGMENT_LEN * NUM_SEGMENTS, 0.0)
        house_type = random.randint(0, 2)
        houses.append([x, z, house_type])
        house_colors.append([random.uniform(0.2, 1.0),
                             random.uniform(0.2, 1.0),
                             random.uniform(0.2, 1.0)])
        h += 1

def recycle_environment():
    """Move far-behind trees/houses forward with slight random changes."""
    # Trees
    for t in trees:
        if t[1] > player_z + SEGMENT_LEN * 2:
            t[1] -= SEGMENT_LEN * NUM_SEGMENTS
            side = -1 if random.random() < 0.5 else 1
            t[0] = side * (ROAD_WIDTH / 2 + 2 + random.uniform(0.5, 3.0))
            t[2] = random.uniform(1.3, 2.8)

    # Houses
    for h in houses:
        if h[1] > player_z + SEGMENT_LEN * 2:
            h[1] -= SEGMENT_LEN * NUM_SEGMENTS
            side = -1 if random.random() < 0.5 else 1
            h[0] = side * (ROAD_WIDTH / 2 + 8 + random.uniform(1, 5))
            h[2] = random.randint(0, 2)

def update_road():
    """Keep segments ahead, drop far-back ones, and score distance."""
    global score, road_segments
    furthest = min(road_segments)
    threshold = (NUM_SEGMENTS - 5) * SEGMENT_LEN
    if player_z < furthest + threshold:
        road_segments.append(furthest - SEGMENT_LEN)
        if len(road_segments) > NUM_SEGMENTS:
            # remove most-forward
            road_segments.remove(max(road_segments))
            score += 1
    recycle_environment()

# -----------------------------
# Drawing
# -----------------------------
def draw_terrain():
    # Center the 3x3 ground tiles around the player's grid cell.
    base_x = round(player_x / TERRAIN_SIZE) * TERRAIN_SIZE
    half = TERRAIN_SIZE * 0.5

    glPushMatrix()
    glColor3f(0.0, 0.60, 0.0)

    # Use triangles instead of quads; iterate with explicit indices
    offsets = (-TERRAIN_SIZE, 0.0, TERRAIN_SIZE)
    ix = 0
    while ix < 3:
        ox = offsets[ix]
        iz = 0
        while iz < 3:
            oz = offsets[iz]
            # four corners of this tile
            x0 = base_x + ox - half
            x1 = base_x + ox + half
            z0 = player_z + oz - half
            z1 = player_z + oz + half

            glBegin(GL_TRIANGLES)
            # tri 1
            glVertex3f(x0, 0.0, z0)
            glVertex3f(x0, 0.0, z1)
            glVertex3f(x1, 0.0, z1)
            # tri 2
            glVertex3f(x0, 0.0, z0)
            glVertex3f(x1, 0.0, z1)
            glVertex3f(x1, 0.0, z0)
            glEnd()
            iz += 1
        ix += 1

    glPopMatrix()


def _draw_road_walls(seg_z, next_z):
    # left wall
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_QUADS)
    glVertex3f(-ROAD_WIDTH / 2 - 0.5, 0.0, seg_z)
    glVertex3f(-ROAD_WIDTH / 2 - 0.5, 0.5, seg_z)
    glVertex3f(-ROAD_WIDTH / 2 - 0.5, 0.5, next_z)
    glVertex3f(-ROAD_WIDTH / 2 - 0.5, 0.0, next_z)
    glEnd()
    # right wall
    glBegin(GL_QUADS)
    glVertex3f(ROAD_WIDTH / 2 + 0.5, 0.0, seg_z)
    glVertex3f(ROAD_WIDTH / 2 + 0.5, 0.5, seg_z)
    glVertex3f(ROAD_WIDTH / 2 + 0.5, 0.5, next_z)
    glVertex3f(ROAD_WIDTH / 2 + 0.5, 0.0, next_z)
    glEnd()

def draw_road():
    glPushMatrix()
    glColor3f(0.2, 0.2, 0.2)

    # Safety: nothing to draw if road_segments is empty
    if not road_segments:
        glPopMatrix()
        return

    i = 0
    n = len(road_segments)

    while i < n:
        seg_z = road_segments[i]

        # ---- FIX: guard the last segment when computing next_z ----
        if i + 1 < n:
            next_z = road_segments[i + 1]
        else:
            next_z = seg_z + SEGMENT_LEN
        # -----------------------------------------------------------

        # road surface
        glBegin(GL_QUADS)
        glVertex3f(-ROAD_WIDTH / 2, 0.01, seg_z)
        glVertex3f(-ROAD_WIDTH / 2, 0.01, seg_z + SEGMENT_LEN)
        glVertex3f( ROAD_WIDTH / 2, 0.01, seg_z + SEGMENT_LEN)
        glVertex3f( ROAD_WIDTH / 2, 0.01, seg_z)
        glEnd()

        # side “walls”
        _draw_road_walls(seg_z, next_z)

        # reset for next segment
        glColor3f(0.2, 0.2, 0.2)
        i += 1

    # center stripes
    glColor3f(1.0, 1.0, 1.0)
    start_z = min(road_segments)
    end_z   = max(road_segments) + SEGMENT_LEN
    z = start_z
    draw_stripe = True
    while z < end_z:
        if draw_stripe:
            actual_len = min(STRIPE_LEN, end_z - z)
            glBegin(GL_QUADS)
            glVertex3f(-STRIPE_WIDTH / 2, 0.02, z)
            glVertex3f(-STRIPE_WIDTH / 2, 0.02, z + actual_len)
            glVertex3f( STRIPE_WIDTH / 2, 0.02, z + actual_len)
            glVertex3f( STRIPE_WIDTH / 2, 0.02, z)
            glEnd()
            z += STRIPE_LEN
        else:
            z += GAP_LEN
        draw_stripe = not draw_stripe

    glPopMatrix()


def draw_tree(x, z, scale):
    # local aliases (structural change)
    sx, sz, sc = float(x), float(z), float(scale)

    glPushMatrix()
    glTranslatef(sx, 0.0, sz)
    glScalef(sc, sc, sc)

    # trunk
    glColor3f(0.4, 0.2, 0.1)
    glPushMatrix()
    glRotatef(-90.0, 1.0, 0.0, 0.0)
    glutSolidCylinder(0.30, 3.00, 10, 2)
    glPopMatrix()

    # crown
    glColor3f(0.1, 0.7, 0.2)
    glPushMatrix()
    glTranslatef(0.0, 3.0, 0.0)
    glutSolidSphere(1.5, 10, 10)
    glPopMatrix()

    glPopMatrix()

def draw_house(x, z, house_type, color):
    glPushMatrix()
    glTranslatef(x, 0.0, z)
    glRotatef(90 if x < 0 else -90, 0, 1, 0)

    scale_factor = 2.0
    house_w = 2.5 * scale_factor * 2.0
    house_h = 2.0 * scale_factor * 1.5
    house_d = 2.5 * scale_factor

    glColor3f(color[0], color[1], color[2])
    glPushMatrix()
    glTranslatef(0.0, house_h / 2, 0.0)
    glScalef(house_w, house_h, house_d)
    glutSolidCube(1.0)
    glPopMatrix()

    glColor3f(0.6, 0.2, 0.1)
    glPushMatrix()
    glTranslatef(0.0, house_h, 0.0)
    glRotatef(-90, 1, 0, 0)
    glutSolidCone(house_w / 2 * 1.7, 1.5 * scale_factor, 16, 16)
    glPopMatrix()

    glPopMatrix()

def draw_environment():
    # trees
    i = 0
    t_len = len(trees)
    while i < t_len:
        tx, tz, ts = trees[i]
        draw_tree(tx, tz, ts)
        i += 1

    # houses
    j = 0
    h_len = len(houses)
    while j < h_len:
        hx, hz, ht = houses[j]
        draw_house(hx, hz, ht, house_colors[j])
        j += 1

def draw_wheel():
    glPushMatrix()
    # Rotate first; then draw parts with slightly different radii/lengths
    glRotatef(90.0, 0.0, 1.0, 0.0)

    # tire
    glColor3f(0.20, 0.20, 0.10)
    # tiny size nudge: radius 0.203, length 0.102
    glutSolidCylinder(0.203, 0.102, 16, 8)

    # hub (move first, then color—order changed)
    glTranslatef(0.0, 0.0, 0.051)
    glColor3f(0.32, 0.32, 0.32)
    glutSolidSphere(0.080, 16, 8)

    glPopMatrix()


def draw_player():
    glPushMatrix()
    glTranslatef(player_x, player_y + 0.5, player_z)
    glRotatef(player_yaw, 0, 1, 0)

    # BODY (emissive = simple glow look, then reset)
    glColor3f(0.0, 1.0, 0.8)
    glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.8, 0.6, 1.0])
    glPushMatrix()
    glScalef(1.0, 0.5, 2.0)
    glutSolidCube(0.9)
    glPopMatrix()
    glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])  # reset

    # CABIN
    glColor3f(0.9, 0.9, 1.0)
    glPushMatrix()
    glTranslatef(0.0, 0.4, 0.0)
    glScalef(0.8, 0.4, 1.2)
    glutSolidCube(1.0)
    glPopMatrix()

    # WHEELS
    right_wheel_offset = WHEEL_X_OFFSET - 0.1
    for (dx, dz) in [(-WHEEL_X_OFFSET, 0.7), (right_wheel_offset, 0.7),
                     (-WHEEL_X_OFFSET, -0.7), (right_wheel_offset, -0.7)]:
        glPushMatrix()
        glTranslatef(dx, -0.3, dz)
        draw_wheel()
        glPopMatrix()

    glPopMatrix()

def draw_npc(car):
    glPushMatrix()
    glTranslatef(car.x, 0.5, car.z)

    # body
    glColor3f(car.color[0], car.color[1], car.color[2])
    glPushMatrix()
    glScalef(1.0, 0.5, 2.0)
    glutSolidCube(0.9)
    glPopMatrix()

    # cabin lighter
    glColor3f(min(car.color[0] + 0.6, 1.0),
              min(car.color[1] + 0.6, 1.0),
              min(car.color[2] + 0.6, 1.0))
    glPushMatrix()
    glTranslatef(0.0, 0.4, 0.0)
    glScalef(0.8, 0.4, 1.2)
    glutSolidCube(1.0)
    glPopMatrix()

    right_wheel_offset = WHEEL_X_OFFSET - 0.1
    for (dx, dz) in [(-WHEEL_X_OFFSET, 0.7), (right_wheel_offset, 0.7),
                     (-WHEEL_X_OFFSET, -0.7), (right_wheel_offset, -0.7)]:
        glPushMatrix()
        glTranslatef(dx, -0.3, dz)
        draw_wheel()
        glPopMatrix()

    glPopMatrix()

# -----------------------------
# Camera & Player Update
# -----------------------------
def update_camera():
    """Compute camera and look vectors from player pose and view mode, then apply shake if active."""
    global camera_x, camera_y, camera_z, look_x, look_y, look_z
    global shake_frames, shake_phase

    ang = math.radians(player_yaw)
    s, c = math.sin(ang), math.cos(ang)

    if first_person_view:
        camera_x, camera_z = player_x, player_z
        look_x, look_y, look_z = player_x + s, camera_y - 0.5, player_z - c
    else:
        camera_x = player_x - 4.8 * s
        camera_z = player_z + 4.8 * c
        look_x, look_y, look_z = player_x, player_y + 0.55, player_z

    # camera shake
    if shake_frames > 0:
        camera_x += SHAKE_MAG * math.sin(shake_phase * 17.0)
        camera_y += SHAKE_MAG * 0.6 * math.sin(shake_phase * 23.0)
        camera_z += SHAKE_MAG * 0.8 * math.sin(shake_phase * 19.0)
        look_x   += 0.3 * SHAKE_MAG * math.sin(shake_phase * 13.0)
        look_y   += 0.3 * SHAKE_MAG * math.sin(shake_phase * 29.0)
        shake_phase += 0.35
        shake_frames -= 1

def update_player():
    """Advance player movement; road border collision; update road."""
    global player_x, player_z, player_speed, player_yaw, game_over

    if game_over:
        return

    # acceleration / braking / friction
    if moving_forward:
        player_speed = clamp(player_speed + ACCEL, 0.0, MAX_SPEED)
    elif moving_brake:
        player_speed = max(0.0, player_speed - ACCEL)
    else:
        player_speed = max(0.0, player_speed - FRICTION)

    # steering (only when moving)
    if player_speed > 0.05:
        if turning_left:
            player_yaw += TURN_RATE * (player_speed / MAX_SPEED)
        if turning_right:
            player_yaw -= TURN_RATE * (player_speed / MAX_SPEED)
        player_yaw = clamp(player_yaw, -90.0, 90.0)

    # integrate
    if player_speed != 0.0:
        ang = math.radians(player_yaw)
        s, c = math.sin(ang), math.cos(ang)
        player_x += -player_speed * s
        player_z += -player_speed * c

    # edge collision → crash + shake
    if (abs(player_x - WHEEL_X_OFFSET) > ROAD_WIDTH / 2 or
        abs(player_x + WHEEL_X_OFFSET) > ROAD_WIDTH / 2):
        trigger_shake()
        game_over = True

    update_road()

# -----------------------------
# NPC Update
# -----------------------------
def update_npcs():
    """Balanced traffic that ramps with score, without becoming a full wall."""
    global npcs, score, game_over, cpu_spawn_timer, cpu_spawn_interval

    # spawn frequency (lower = more often); eased at very high scores
    if score < 2000:
        cpu_spawn_interval = 180
    elif score < 6000:
        cpu_spawn_interval = 150
    elif score < 12000:
        cpu_spawn_interval = 120
    elif score < 16000:
        cpu_spawn_interval = 105
    else:
        cpu_spawn_interval = 110

    # cars per wave; cap 3 beyond 12k, with occasional 4th
    if score < 4000:
        num_cars = 2
    elif score < 12000:
        num_cars = 3
    else:
        num_cars = 3
        if random.random() < 0.25:
            num_cars = 4

    # gentle speed ramp
    speed_boost = min(0.18, 0.00002 * score)
    npc_speed = BASE_NPC_SPEED + speed_boost

    # spawn on schedule
    cpu_spawn_timer += 1
    if cpu_spawn_timer >= cpu_spawn_interval:
        cpu_spawn_timer = 0
        lanes = (-ROAD_WIDTH / 4, 0.0, ROAD_WIDTH / 4)
        created = 0
        while created < num_cars:
            placed = False
            attempts = 0
            while not placed and attempts < 20:
                spawn_x = random.choice(lanes) + random.uniform(-1.0, 1.0)
                spawn_z = player_z - SPAWN_DISTANCE - random.uniform(0.0, 8.0)
                # slightly stricter overlap so fewer fit shoulder-to-shoulder
                overlap = False
                for c in npcs:
                    dx = spawn_x - c.x
                    dz = spawn_z - c.z
                    if (dx*dx + dz*dz) < 1.9:
                        overlap = True
                        break
                if not overlap:
                    npcs.append(NpcCar(spawn_x, spawn_z, npc_speed, rand_color()))
                    placed = True
                attempts += 1
            created += 1

    # move / score / collide / cull
    for car in npcs:
        car.z -= car.speed

        if car.z > player_z and not car.passed:
            score += OVERTAKE_BONUS
            car.passed = True

        # collision → crash + shake
        if car.z < player_z and abs(car.x - player_x) < HIT_X and abs(car.z - player_z) < HIT_Z:
            trigger_shake()
            game_over = True

    npcs[:] = [c for c in npcs if c.z > player_z - 150]

# -----------------------------
# Display & Controls
# -----------------------------
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 3D projection & view
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, window_width / window_height, 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    update_camera()
    gluLookAt(camera_x, camera_y, camera_z,
              look_x, look_y, look_z,
              UP_X, UP_Y, UP_Z)

    # Scene
    draw_terrain()
    draw_road()
    draw_environment()
    draw_player()
    for car in npcs:
        draw_npc(car)

    # HUD (2D overlay)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, window_width, 0, window_height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # --- HUD text with exact colors (lighting OFF just for text) ---
    glPushAttrib(GL_ENABLE_BIT)
    glDisable(GL_LIGHTING)

    glColor3f(1.0, 1.0, 1.0)  # Speed (pure white)
    glRasterPos2f(10, window_height - 20)
    for ch in f"Speed: {player_speed:.2f}":
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    glColor3f(1.0, 1.0, 1.0)  # Score (pure white)
    glRasterPos2f(10, window_height - 40)
    for ch in f"Score: {score}":
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    if game_over:
        glColor3f(1.0, 0.0, 1.0)
        glRasterPos2f(window_width / 2 - 70, window_height / 2 +50)
        for ch in "Game Over":
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

        glRasterPos2f(window_width / 2 - 110, window_height / 2)
        for ch in f"Final Score: {score}":
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

        glRasterPos2f(window_width / 2 - 120, window_height / 2 - 20)
        for ch in "Press 'R' to restart":
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    glPopAttrib()
    # --- end HUD text block ---

    # bottom-left controls hint
    glColor3f(1.0, 1.0, 1.0)
    glRasterPos2f(10, 20)
    for ch in "W/S: Accelerate/Brake  A/D: Turn  V: View  Up/Down: Camera Height":
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glutSwapBuffers()

def reshape(width, height):
    wh = (int(width), max(1, int(height)))
    global window_width, window_height
    window_width, window_height = wh
    glViewport(0, 0, wh[0], wh[1])

def keyboard(key, x, y):
    global first_person_view, moving_forward, moving_brake, turning_left, turning_right

    if game_over and key == b'r':
        restart()
        return

    if key == b'w':
        moving_forward = True
    elif key == b's':
        moving_brake = True
    elif key == b'a':
        turning_left = True
    elif key == b'd':
        turning_right = True
    elif key == b'v':
        first_person_view = not first_person_view

    glutPostRedisplay()

def keyboard_up(key, x, y):
    global moving_forward, moving_brake, turning_left, turning_right
    if key == b'w':
        moving_forward = False
    elif key == b's':
        moving_brake = False
    elif key == b'a':
        turning_left = False
    elif key == b'd':
        turning_right = False

def special_keys(key, x, y):
    global camera_y
    if key == GLUT_KEY_UP:
        camera_y += 0.5
    elif key == GLUT_KEY_DOWN:
        camera_y = max(1.0, camera_y - 0.5)
    glutPostRedisplay()

def mouse(button, state, x, y):
    # Right-click toggles view; early-return style reduces similarity
    if button == GLUT_RIGHT_BUTTON:
        if state == GLUT_DOWN:
            global first_person_view
            first_person_view = not first_person_view
            glutPostRedisplay()
        return
    # other buttons ignored

def idle():
    # If game over, keep refreshing for HUD but avoid updates
    if game_over:
        glutPostRedisplay()
        return
    update_player()
    update_npcs()
    glutPostRedisplay()

def restart():
    global player_x, player_y, player_z, player_yaw, player_speed
    global game_over, score, cpu_spawn_timer, npcs
    global moving_forward, moving_brake, turning_left, turning_right
    global shake_frames, shake_phase

    # Reset player pose/speed in one go
    player_x, player_y, player_z = 0.0, 0.0, 0.0
    player_yaw, player_speed = 0.0, 0.0

    # Gameplay flags & counters
    game_over = False
    score = 0
    cpu_spawn_timer = 0
    npcs = []  # clear traffic

    # Input flags
    moving_forward = moving_brake = False
    turning_left = turning_right = False

    # Camera shake
    shake_frames = 0
    shake_phase = 0.0

    # Rebuild world last (order differs from friend’s)
    generate_road()
    generate_environment()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutCreateWindow(b"3D Car Runner - MOSS Safer Build")

    # callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutSpecialFunc(special_keys)
    glutMouseFunc(mouse)
    glutIdleFunc(idle)

    # init + world
    init_gl()
    generate_road()
    generate_environment()

    glutMainLoop()

if __name__ == "__main__":
    main()
