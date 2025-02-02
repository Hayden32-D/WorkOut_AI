import numpy as np

def calculate_angle(A, B, C):
    """
    Calculate the angle at point B given three points (A, B, C).
    A, B, C should be (x, y) coordinates.
    Returns angle in degrees.
    """
    a = np.linalg.norm(np.array(B) - np.array(C))  # Distance BC
    b = np.linalg.norm(np.array(A) - np.array(C))  # Distance AC
    c = np.linalg.norm(np.array(A) - np.array(B))  # Distance AB

    # Apply cosine rule to find angle at B
    angle = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
    return np.degrees(angle)  # Convert radians to degrees


def get_knee_angle(left_hip, left_knee, left_ankle):
    """Calculate knee angle for squat form."""
    return calculate_angle(left_hip, left_knee, left_ankle)

def get_hip_angle(left_shoulder, left_hip, left_knee):
    """Calculate hip angle to check squat depth and hip mobility."""
    return calculate_angle(left_shoulder, left_hip, left_knee)


class SquatFormChecker:
    def __init__(self):
        self.state = "STANDING"

    def check_squat(self, knee_angle, hip_angle):
        """
        Evaluates squat form and deducts points for incorrect form.
        """
        score = 100  # Start with a perfect score

        # Determine squat phase
        if knee_angle > 160 and hip_angle > 160:
            self.state = "STANDING"
        elif self.state == "STANDING" and knee_angle < 120 and hip_angle < 120:
            self.state = "LOWERING"
        elif 90 <= knee_angle <= 110 and 90 <= hip_angle <= 110:
            self.state = "SQUAT_POSITION"
        elif self.state == "SQUAT_POSITION" and knee_angle > 140 and hip_angle > 120:
            self.state = "COMING_UP"

        # Only deduct points in 'SQUAT_POSITION' state
        if self.state == "SQUAT_POSITION":
            if knee_angle < 70 and knee_angle > 50:
                score -= 70 - knee_angle 
            if knee_angle > 140 and knee_angle < 180:
                score -= knee_angle - 140
            if hip_angle < 60:
                score -= 60 - hip_angle 
            if hip_angle > 150 and hip_angle < 170:
                score -= hip_angle - 150  
            
        score = int(score)
        return score, self.state
