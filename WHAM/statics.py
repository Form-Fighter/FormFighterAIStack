import numpy as np


# Guard functions
def check_head_stability(joints_sequence, threshold_degrees=30):
    """
    Checks if the head direction remains stable within a threshold in degrees.

    Args:
        joints_sequence: A numpy array of shape (frames, joints, 3), where each frame represents the joint positions.
        threshold_degrees: The allowed angle change in degrees for the head direction.

    Returns:
        True if the head remains stable within the threshold, False otherwise.
    """
    # Calculate the initial forward direction (based on the first frame)
    initial_joints = joints_sequence[0]

    initial_forward_direction = calculate_forward_direction(initial_joints)

    for joints in joints_sequence[1:]:

        current_forward_direction = calculate_forward_direction(joints)

        # Calculate the angle between the initial and current forward direction
        dot_product = np.dot(initial_forward_direction, current_forward_direction)
        angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_degrees = np.degrees(angle_radians)
        print(angle_degrees)
        # Check if the angle exceeds the threshold
        if angle_degrees > threshold_degrees:
            return False  # Head turned significantly

    return True  # Head remained stable within the threshold


def check_leg_to_shoulder_width(joints, threshold=0.3):
    """
    Checks if the horizontal distance between the legs is close to the horizontal distance
    between the shoulders within a specified threshold.

    Args:
        joints: A numpy array of shape (joints, 3), where each row represents the (x, y, z) coordinates of a joint.
        threshold: The allowed difference between shoulder width and leg width.

    Returns:
        True if legs are approximately shoulder-width apart, False otherwise.
    """
    # Shoulder positions
    left_shoulder = joints[5]
    right_shoulder = joints[6]

    # Leg positions
    left_leg = joints[15]
    right_leg = joints[16]

    # Calculate the horizontal (XZ) distance between shoulders
    shoulder_width = np.linalg.norm(left_shoulder[[0, 2]] - right_shoulder[[0, 2]])

    # Calculate the horizontal (XZ) distance between legs
    leg_width = np.linalg.norm(left_leg[[0, 2]] - right_leg[[0, 2]])
    # Check if the leg width is within the threshold range of shoulder width
    if abs(leg_width - shoulder_width) <= threshold:
        return True
    else:
        return False


def check_back_leg_angle(joints, stance):
    """
    Checks if the back leg is at approximately a 45-degree angle with the ground (assumed to be the ZX plane).

    Args:
        joints: A numpy array of shape (joints, 3), where each row represents the (x, y, z) coordinates of a joint.
        stance: A string, either 'orthodox' or 'southpaw', indicating the stance of the person.

    Returns:
        The angle (in degrees) between the back leg (foot-to-knee vector) and the ground plane.
    """
    # Select back leg joints based on stance
    if stance == 'orthodox':
        foot_index, knee_index = 16, 14  # Right leg
    else:
        foot_index, knee_index = 15, 13  # Left leg

    # Get coordinates of the foot and knee
    foot = joints[foot_index]
    knee = joints[knee_index]

    # Calculate the foot-to-knee vector
    leg_vector = knee - foot

    # Project leg_vector onto the ZX plane (ignore y component)
    leg_vector_zx = np.array([leg_vector[0], 0, leg_vector[2]])

    # Calculate the angle between leg_vector and the ground plane
    angle_radians = np.arccos(np.dot(leg_vector_zx, leg_vector) /
                              (np.linalg.norm(leg_vector_zx) * np.linalg.norm(leg_vector)))
    angle_degrees = np.degrees(angle_radians)

    # Check if angle is close to 45 degrees
    target_angle = 45  # Target angle in degrees
    tolerance = 10      # Tolerance in degrees

    if abs(angle_degrees - target_angle) <= tolerance:
        return True  # Leg is at approximately 45 degrees
    else:
        return False  # Leg is not at approximately 45 degrees


def check_guard_hands_above_shoulders(joints):
    """
    Checks if both hands (left: index 9, right: index 23) are above the height of the respective shoulders
    (left shoulder: index 5, right shoulder: index 6).

    Args:
        joints: A numpy array of shape (joints, 3), where each row represents the (x, y, z) coordinates of a joint.

    Returns:
        A score indicating if both hands are above shoulder height (1 if both hands are above, 0 otherwise).
    """
    # Left hand and shoulder height comparison
    left_hand_height = joints[9][1]   # y-coordinate of the left hand
    left_shoulder_height = joints[5][1]  # y-coordinate of the left shoulder

    # Right hand and shoulder height comparison
    right_hand_height = joints[23][1]  # y-coordinate of the right hand
    right_shoulder_height = joints[6][1]  # y-coordinate of the right shoulder

    # Check if both hands are above their respective shoulders
    if left_hand_height > left_shoulder_height and right_hand_height > right_shoulder_height:
        return 1  # Both hands are in guard position
    else:
        return 0  # Hands are not fully in guard position


def detect_stance(joints, forward_direction):
    """
    Detect if the person is in a southpaw or orthodox stance based on the position of the body joints,
    taking into account the direction the person is facing.

    Args:
        joints: A numpy array of shape (joints, 3), where each row represents the (x, y, z) coordinates of a joint.
        forward_direction: A 3D vector representing the forward direction.

    Returns:
        'southpaw' if the left side of the body is more forward, 'orthodox' if the right side is more forward.
    """
    # # Determine if the person is facing forward or backward based on the nose (index 0) position
    # head_direction = joints[1] - joints[0]  # Nose to left eye vector
    # facing_forward = np.dot(forward_direction, head_direction) > 0
    #
    # # If the person is facing backward, reverse the forward direction
    # if not facing_forward:
    #     forward_direction = -forward_direction

    # Right side joints
    right_side_joints = [joints[i] for i in [16, 14, 12, 19, 6, 8, 23]]

    # Left side joints
    left_side_joints = [joints[i] for i in [15, 13, 11, 20, 5, 7, 9]]

    # Project right side joints onto the forward direction
    right_side_distances = [np.dot(joint - joints[0], forward_direction) for joint in right_side_joints]

    # Project left side joints onto the forward direction
    left_side_distances = [np.dot(joint - joints[0], forward_direction) for joint in left_side_joints]

    # Calculate the average forward distance for both sides
    avg_right_distance = np.mean(right_side_distances)
    avg_left_distance = np.mean(left_side_distances)

    # Determine the stance based on which side is more forward
    if avg_left_distance > avg_right_distance:
        return 'orthodox'
    else:
        return 'southpaw'


def is_in_guard_phase(joints):
    """
    Determines if the left hand (index 9) is in the guard position.

    Args:
        joints: A numpy array of shape (joints, 3), where each row represents the (x, y, z) coordinates of a joint.

    Returns:
        True if the hand is in the guard phase, otherwise False.
    """
    left_hand = joints[9]  # (x, y, z) of the left hand
    left_eye = joints[1]  # (x, y, z) of the left eye
    left_ear = joints[3]  # (x, y, z) of the left ear

    # Calculate vertical (y-axis) distance to eye and ear
    hand_eye_distance = abs(left_hand[1] - left_eye[1])
    hand_ear_distance = abs(left_hand[1] - left_ear[1])

    # Calculate horizontal distance (x-axis and z-axis) between hand and face
    horizontal_distance_to_eye = np.linalg.norm(left_hand[[0, 2]] - left_eye[[0, 2]])
    horizontal_distance_to_ear = np.linalg.norm(left_hand[[0, 2]] - left_ear[[0, 2]])

    # Threshold values (you may need to adjust these based on data)
    vertical_threshold = 0.35  # How close the hand should be vertically to the eye/ear
    horizontal_threshold = 0.35  # How close the hand should be horizontally

    # Check if the hand is close to the eye and ear both vertically and horizontally
    if (hand_eye_distance < vertical_threshold and
            hand_ear_distance < vertical_threshold and
            horizontal_distance_to_eye < horizontal_threshold and
            horizontal_distance_to_ear < horizontal_threshold):
        return True
    return False


# Statics
def calculate_forward_direction(joints):
    """
    Calculate the forward direction as the vector perpendicular to the line between the eyes and passing through the nose.

    Args:
        joints: A numpy array of shape (joints, 3), where each row represents the (x, y, z) coordinates of a joint.

    Returns:
        A normalized 3D vector representing the forward direction.
    """
    left_eye = joints[1]
    right_eye = joints[2]
    nose = joints[0]

    # Calculate the vector from left eye to right eye
    eye_vector = right_eye - left_eye

    # Calculate the vector from the nose to the midpoint between the eyes
    eye_midpoint = (left_eye + right_eye) / 2
    forward_vector = nose - eye_midpoint

    # Find the vector perpendicular to the eye_vector and forward_vector using cross product
    forward_direction = np.cross(eye_vector, forward_vector)

    # Normalize the forward direction vector
    forward_direction_normalized = forward_direction / np.linalg.norm(forward_direction)

    return forward_direction_normalized


def calculate_acceleration(joints_sequence, joint_index, forward_direction):
    """
    Calculate the average signed acceleration of a joint in the forward direction across a sequence of frames.

    Args:
        joints_sequence: A numpy array of shape (frames, joints, 3), representing the joint positions over time.
        joint_index: The index of the joint for which to calculate acceleration (e.g., left hand at index 9).
        forward_direction: A 3D vector representing the forward direction.

    Returns:
        The average signed acceleration of the joint in the forward direction.
    """
    velocities = []
    accelerations = []

    # Calculate signed velocities in the forward direction for each frame
    for i in range(1, len(joints_sequence)):

        # Current and previous positions of the joint
        prev_position = joints_sequence[i - 1, joint_index]
        curr_position = joints_sequence[i, joint_index]

        # Calculate displacement and signed velocity in the forward direction
        displacement = curr_position - prev_position
        velocity = np.dot(displacement, forward_direction)  # Dot product gives signed velocity
        velocities.append(velocity)

    # Calculate signed accelerations from velocities
    for i in range(1, len(velocities)):
        # Difference in velocity between consecutive frames
        acceleration = velocities[i] - velocities[i - 1]
        accelerations.append(acceleration)

    # Return the average signed acceleration
    if accelerations:
        return np.mean(accelerations)
    else:
        return 0  # No movement detected