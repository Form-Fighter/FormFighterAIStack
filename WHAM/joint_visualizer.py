import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


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


class JointMotionVisualizer:
    def __init__(self):
        self.joints_sequence = None
        self.current_joint = 0
        self.fig = None
        self.ax = None
        self.scatter = None
        self.highlight = None
        self.text_label = None
        self.trajectory = None
        self.acceleration_list = []
        self.acceleration_window = 7

    def animate_joints(self, joints_sequence: np.ndarray, interval: int=50, highlight_joint=None):
        """
        Animate the joint movement sequence with optional joint highlighting and connections between joints.

        Args:
            joints_sequence: numpy array of shape (frames, joints, 3)
            interval: milliseconds between frames
            highlight_joint: joint index to highlight and track
        """
        self.joints_sequence = joints_sequence
        self.current_joint = highlight_joint if highlight_joint is not None else 0

        # Create figure
        self.fig = plt.figure(figsize=(15, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize scatter plot
        frame_data = self.joints_sequence[0]
        self.scatter = self.ax.scatter(frame_data[:, 0],
                                       frame_data[:, 1],
                                       frame_data[:, 2],
                                       c='blue',
                                       s=50)

        # Add text labels next to each joint
        self.text_labels = []
        for i, joint in enumerate(frame_data):
            label = self.ax.text(joint[0], joint[1], joint[2], f'{i}', fontsize=12, color='black')
            self.text_labels.append(label)

        # Define connections between adjacent body parts based on your instructions
        self.connections = [
            # Spine
            (11, 12), (12, 19), (19, 20),
            # Head and face
            (0, 1), (1, 2), (0,2),  # Nose to eyes
            (3, 4),  # Ears
            (29, 30),  # Spine end to top of head
            # Right leg
            (16, 14), (14, 12), (12, 19),
            # Left leg
            (15, 13), (13, 11), (11, 20),
            # Right leg alternate connections
            (17, 18), (18, 19), (22, 21),
            # Left arm
            (5, 7), (7, 9),  # Left shoulder to elbow to hand
            # Right arm
            (6, 8), (8, 23),  # Right shoulder to elbow to hand
            (6, 19), (5, 20),  # Right hip to shoulder, left hip to shoulder
            (5, 29), (6, 29),  # Spine to shoulders
            (0, 29), (0, 30)  # Head to spine, head to top of head
        ]
        self.lines = []
        for start, end in self.connections:
            line, = self.ax.plot([frame_data[start, 0], frame_data[end, 0]],
                                 [frame_data[start, 1], frame_data[end, 1]],
                                 [frame_data[start, 2], frame_data[end, 2]], 'b-', lw=2)
            self.lines.append(line)

        if highlight_joint is not None:
            self.highlight = self.ax.scatter(frame_data[highlight_joint, 0],
                                             frame_data[highlight_joint, 1],
                                             frame_data[highlight_joint, 2],
                                             c='red',
                                             s=100)

            # Initialize trajectory line
            self.trajectory = self.ax.plot([], [], [], 'r-', alpha=0.5)[0]

        # Add text label for frame and highlighted joint
        self.text_label = self.ax.text2D(0.05, 0.95,
                                         f"Frame: 0/{len(joints_sequence) - 1}\n" +
                                         f"Highlighted Joint: {highlight_joint}",
                                         transform=self.ax.transAxes)

        # Set axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set consistent axis limits
        all_x = joints_sequence[:, :, 0].flatten()
        all_y = joints_sequence[:, :, 1].flatten()
        all_z = joints_sequence[:, :, 2].flatten()

        x_margin = (all_x.max() - all_x.min()) * 0.1
        y_margin = (all_y.max() - all_y.min()) * 0.1
        z_margin = (all_z.max() - all_z.min()) * 0.1

        self.ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        self.ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
        self.ax.set_zlim(all_z.min() - z_margin, all_z.max() + z_margin)

        # Create animation
        anim = FuncAnimation(self.fig,
                             self._update_frame,
                             frames=len(joints_sequence),
                             interval=interval,
                             blit=False)

        plt.show()

    def _update_frame(self, frame):
        """Update function for animation."""
        frame_data = self.joints_sequence[frame]

        # Update all joints
        self.scatter._offsets3d = (frame_data[:, 0],
                                   frame_data[:, 1],
                                   frame_data[:, 2])

        # Remove old text labels
        for label in self.text_labels:
            label.remove()

        # Redraw new text labels for this frame
        self.text_labels = []
        for i, joint in enumerate(frame_data):
            label = self.ax.text(joint[0], joint[1], joint[2], f'{i}', fontsize=12, color='black')
            self.text_labels.append(label)

        # Update the lines connecting joints
        for i, (start, end) in enumerate(self.connections):
            self.lines[i].set_data([frame_data[start, 0], frame_data[end, 0]],
                                   [frame_data[start, 1], frame_data[end, 1]])
            self.lines[i].set_3d_properties([frame_data[start, 2], frame_data[end, 2]])

        if self.highlight is not None:
            # Update highlighted joint
            self.highlight._offsets3d = (frame_data[self.current_joint:self.current_joint + 1, 0],
                                         frame_data[self.current_joint:self.current_joint + 1, 1],
                                         frame_data[self.current_joint:self.current_joint + 1, 2])

            # Update trajectory
            trajectory_data = self.joints_sequence[:frame + 1, self.current_joint]
            self.trajectory.set_data(trajectory_data[:, 0], trajectory_data[:, 1])
            self.trajectory.set_3d_properties(trajectory_data[:, 2])

        forward_direction = calculate_forward_direction(frame_data)

        phase = ''
        acceleration = 'N/A'
        if frame > self.acceleration_window:
            acceleration = calculate_acceleration(self.joints_sequence[frame-self.acceleration_window:frame], 9, forward_direction)
            self.acceleration_list.append(acceleration)
            averaged_acceleration = np.mean(self.acceleration_list)
            # if acceleration > 0.01:
            #     phase = "Extension"
            # elif acceleration < -0.01:
            #     phase = "Retraction"
            # else:
            #     phase = ""
            if averaged_acceleration > 0:
                phase = "Extension"
            elif averaged_acceleration < 0:
                phase = "Retraction"
            else:
                phase = ""

        stance = detect_stance(frame_data, forward_direction)
        # Update text label for frame and highlighted joint
        is_in_guard_check = is_in_guard_phase(frame_data)
        is_in_guard = "Guard position" if is_in_guard_check is True else phase

        guard_feedback = 'Guard feedback:'

        if is_in_guard_check:
            # Hands above shoulders check
            if check_guard_hands_above_shoulders(frame_data):
                guard_feedback += 'Hands above shoulders'
            else:
                guard_feedback += 'Hands not above shoulders'

            # Back leg at 45 degrees check
            back_leg_check = check_back_leg_angle(frame_data, stance)
            if back_leg_check:
                guard_feedback += ', Back leg at 45 degrees'
            else:
                guard_feedback += ', Back leg not at 45 degrees'

            # Leg to shoulder width check
            if check_leg_to_shoulder_width(frame_data):
                guard_feedback += ', Legs at shoulder width'
            else:
                guard_feedback += ', Legs not at shoulder width'

            # Head stability check
            if check_head_stability(self.joints_sequence):
                guard_feedback += ', Head stable'
            else:
                guard_feedback += ', Head not stable'

        else:
            guard_feedback += 'Not in guard position'

        self.text_label.set_text(f"Frame: {frame}/{len(self.joints_sequence) - 1}\n" +
                                 f"Highlighted Joint: {self.current_joint}\n" +
                                 f"Position: {is_in_guard}\n" +
                                 f"Acceleration: {acceleration}\n"
                                 f"Stance: {stance} \n" +
                                 f"{guard_feedback}\n")

        # Back leg at 45 degrees check
        back_leg_check = check_back_leg_angle(frame_data, stance)


        if frame >= len(self.joints_sequence)-1:
            self.acceleration_list = []


class JointIdentifier:
    def __init__(self):
        self.current_joint = 0
        self.fig = None
        self.ax = None
        self.joints = None
        self.scatter = None
        self.highlight = None
        self.text_label = None

    def visualize_joints_interactive(self, joints):
        """Same as before but for single frame reference."""
        self.joints = joints
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.scatter = self.ax.scatter(joints[:, 0],
                                       joints[:, 1],
                                       joints[:, 2],
                                       c='blue',
                                       s=50)

        self.highlight = self.ax.scatter(joints[self.current_joint, 0],
                                         joints[self.current_joint, 1],
                                         joints[self.current_joint, 2],
                                         c='red',
                                         s=100)

        self.text_label = self.ax.text2D(0.05, 0.95,
                                         f"Joint Index: {self.current_joint}\n" +
                                         f"Coordinates: ({joints[self.current_joint, 0]:.2f}, " +
                                         f"{joints[self.current_joint, 1]:.2f}, " +
                                         f"{joints[self.current_joint, 2]:.2f})",
                                         transform=self.ax.transAxes)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Press any key to cycle through joints\nClose window when done')

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.show()

    def _on_key_press(self, event):
        """Handle keyboard press to cycle through joints."""
        self.current_joint = (self.current_joint + 1) % len(self.joints)

        self.highlight.remove()
        self.highlight = self.ax.scatter(self.joints[self.current_joint, 0],
                                         self.joints[self.current_joint, 1],
                                         self.joints[self.current_joint, 2],
                                         c='red',
                                         s=100)

        self.text_label.set_text(f"Joint Index: {self.current_joint}\n" +
                                 f"Coordinates: ({self.joints[self.current_joint, 0]:.2f}, " +
                                 f"{self.joints[self.current_joint, 1]:.2f}, " +
                                 f"{self.joints[self.current_joint, 2]:.2f})")

        self.fig.canvas.draw_idle()


# Example usage:
if __name__ == "__main__":
    visualizer = JointMotionVisualizer()
    joints_sequence = np.load("joints.npy")
    # Interactive joint selection and animation
    visualizer.animate_joints(joints_sequence, interval=50, highlight_joint=9)

