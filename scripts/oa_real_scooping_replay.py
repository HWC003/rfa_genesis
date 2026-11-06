"""
Replay a soybean CSV through the two arm action controllers.

Both modes go through a fixed intermediate START_POSE before the CSV start:
    measured (normally all zeros) -> START_POSE -> CSV first row -> CSV replay

--preview-approach displays a virtual zero -> START_POSE -> CSV-start motion in MoveIt/RViz.
It does not send action goals or move hardware.
--execute: both arms -> START_POSE; right arm -> CSV start; wait 2 s; left gripper
quick open, hold 3 s, quick close; left arm -> CSV start; both arms replay the CSV together.

Smooth start: the joint CSVs start mid-motion (up to ~6 rad/s). The first --blend-window s (default 0.3) are
retimed over --blend-duration s (default 0.7) with a quintic time scaling (trajectory_smoothing.py): same
joint path, starting from rest. --blend-window 0 replays the CSV unchanged.

Payload: /left_payload_attached (std_msgs/Bool, transient local) is published False before the gripper opens
and True after it closes on the bowl, so the feeding bringup's gravity compensation carries the bowl.

Recording: from the scoop start (time 0 of the replay) to 1 s after its end, /joint_states is saved to
--record-dir/<csv stem>_<date_time>/measured.csv next to commanded.csv (the trajectory actually sent).
Both use the CSV's column names; time_from_start_s is relative to the replay start.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from action_msgs.msg import GoalStatus

from trajectory_smoothing import smooth_start, peak_velocity

# Intermediate start pose (rad), joints 1-7, reached before moving to the CSV start.
START_POSE = {
    "left": (0.785, 0.0, 0.0, 1.570, 0.0, 0.0, 0.0),
    "right": (-0.785, 0.0, 0.0, 1.570, 0.0, 0.0, 0.0),
}

GRIPPER_OPEN = 0.785    # rad, from openarm_bowl_grasp.yaml
GRIPPER_CLOSED = 0.0

ARM_JOINTS = [f"openarm_{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
RECORD_DIR = Path(__file__).resolve().parent.parent / "real_runs"


def start_pose_positions():
    """START_POSE as {joint_name: angle} for both arms."""
    return {
        f"openarm_{side}_joint{i}": angle
        for side, angles in START_POSE.items()
        for i, angle in enumerate(angles, start=1)
    }


class ScoopingReplay(Node):
    def __init__(self):
        super().__init__('scooping_replay')

        self.left_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/left_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.right_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/right_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.current_positions = {}
        self.joint_received_at = {}

        self.left_gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/left_gripper_controller/follow_joint_trajectory'
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile_sensor_data
        )

        # Bowl payload for the feeding bringup's gravity compensation (latched, as xarm_compat publishes it).
        self.payload_pub = self.create_publisher(
            Bool, '/left_payload_attached',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
        )

        # /joint_states recording: rows of (time from replay start, 14 positions, 14 velocities, 14 efforts).
        self.recording = False
        self.record_start_ns = 0
        self.recorded = []

    def set_payload(self, attached):
        self.payload_pub.publish(Bool(data=bool(attached)))
        self.get_logger().info(f"/left_payload_attached -> {bool(attached)}")

    def joint_state_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            self.current_positions[name] = position
            self.joint_received_at[name] = time.monotonic()
        if self.recording:
            index = {name: i for i, name in enumerate(msg.name)}
            if all(name in index for name in ARM_JOINTS):
                stamp = Time.from_msg(msg.header.stamp).nanoseconds or self.get_clock().now().nanoseconds
                if stamp < self.record_start_ns:
                    return          # goals sent but the scoop has not started yet
                def column(values):
                    return [values[index[n]] if len(values) == len(msg.name) else float('nan') for n in ARM_JOINTS]
                self.recorded.append([(stamp - self.record_start_ns) / 1e9] + column(msg.position)
                                     + column(msg.velocity) + column(msg.effort))

    def save_recording(self, run_dir):
        """Write the recorded /joint_states to run_dir/measured.csv."""
        header = (["time_from_start_s"] + ARM_JOINTS + [f"velocity_{n}" for n in ARM_JOINTS]
                  + [f"effort_{n}" for n in ARM_JOINTS])
        with (run_dir / "measured.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.recorded)
        self.get_logger().info(f"Saved {len(self.recorded)} /joint_states samples to {run_dir / 'measured.csv'}")

    def read_current_positions(self, timeout_s=5.0):
        required_names = [
            f"openarm_{side}_joint{i}"
            for side in ["left", "right"]
            for i in range(1, 8)
        ]

        deadline = time.monotonic() + timeout_s

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if all(name in self.current_positions and
                   time.monotonic() - self.joint_received_at[name] < 0.5
                   for name in required_names):
                position = {
                    name: self.current_positions[name]
                    for name in required_names
                }

                if not all(math.isfinite(pos) for pos in position.values()):
                    raise ValueError("Received invalid joint positions")

                return position

        raise RuntimeError("Did not receive all 14 arm joints on /joint_states")

    def check_connections(self):
        left_ready = self.left_client.wait_for_server(timeout_sec=5.0)
        right_ready = self.right_client.wait_for_server(timeout_sec=5.0)

        self.get_logger().info(f'Left arm action server ready: {left_ready}')
        self.get_logger().info(f'Right arm action server ready: {right_ready}')
        if not left_ready or not right_ready:
            raise RuntimeError("Both arm action servers must be available")

    def load_trajectory(self, csv_path):
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        if not rows:
            raise ValueError("The trajectory CSV is empty.")

        self.get_logger().info(f"Loaded trajectory: {csv_path.name}")
        self.get_logger().info(f"Number of points: {len(rows)}")
        self.get_logger().info(
            f"Final timestamp: {rows[-1]['time_from_start_s']} seconds"
        )

        return rows

    def smooth_rows(self, rows, window, duration):
        """Retime the first `window` s of the CSV to start from rest (trajectory_smoothing.smooth_start)."""
        times = np.array([float(row["time_from_start_s"]) for row in rows])
        q = np.array([[float(row[name]) for name in ARM_JOINTS] for row in rows])
        new_times, new_q = smooth_start(times, q, window, duration)
        if window > 0:
            self.get_logger().info(
                f"Smooth start: first {window:.3f} s retimed over {duration:.3f} s; peak joint speed "
                f"{peak_velocity(times, q):.2f} -> {peak_velocity(new_times, new_q):.2f} rad/s; "
                f"replay {times[-1]:.2f} -> {new_times[-1]:.2f} s"
            )
        return [dict(time_from_start_s=t, **dict(zip(ARM_JOINTS, joints))) for t, joints in zip(new_times, new_q)]

    @staticmethod
    def save_commanded(rows, run_dir):
        with (run_dir / "commanded.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["time_from_start_s"] + ARM_JOINTS)
            writer.writeheader()
            writer.writerows(rows)

    def build_goal(self, side, rows):
        goal = FollowJointTrajectory.Goal()

        joint_names=[
            f"openarm_{side}_joint{i}"
            for i in range(1, 8)
        ]
        goal.trajectory.joint_names = joint_names

        previous_time_ns = -1

        for row in rows:
            positions = [
                float(row[name])
                for name in joint_names
            ]
            time_s = float(row["time_from_start_s"])

            if not all(math.isfinite(pos) for pos in positions):
                raise ValueError(f"Non-finite joint positions found in row: {row}")

            if not math.isfinite(time_s) or time_s < 0:
                raise ValueError(f"Non-finite time_from_start_s found in row: {row}")

            time_ns = round(time_s * 1e9)

            if time_ns <= previous_time_ns:
                raise ValueError("Timestamps must increase")

            point = JointTrajectoryPoint()
            point.positions = positions 
            point.time_from_start = Duration(nanoseconds=time_ns).to_msg()
            goal.trajectory.points.append(point)
            previous_time_ns = time_ns

        return goal

    def build_approach(self, replay_goal, current_positions, duration_s=10.0):
        """Quintic move from the given angles to the CSV start."""
        names = list(replay_goal.trajectory.joint_names)
        return self.build_move(
            names, current_positions,
            dict(zip(names, replay_goal.trajectory.points[0].positions)), duration_s,
        )

    def build_start_pose_move(self, side, current_positions, duration_s=10.0):
        """Quintic move for one arm from the given angles to START_POSE."""
        names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        return self.build_move(names, current_positions, start_pose_positions(), duration_s)

    def build_move(self, joint_names, start_positions, target_positions, duration_s=10.0):
        """Sample a quintic joint spline from start to target angles at 200 Hz.

        Both endpoint velocities and accelerations are zero. This is joint-space
        interpolation; it does not perform collision checking or motion planning.
        """
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Move duration must be finite and positive")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        starts = [start_positions[name] for name in goal.trajectory.joint_names]
        targets = [target_positions[name] for name in goal.trajectory.joint_names]
        changes = [target - start for start, target in zip(starts, targets)]
        steps = math.ceil(duration_s / 0.005)

        for step in range(steps + 1):
            u = step / steps
            blend = 10*u**3 - 15*u**4 + 6*u**5
            speed = (30*u**2 - 60*u**3 + 30*u**4) / duration_s
            acceleration = (60*u - 180*u**2 + 120*u**3) / duration_s**2

            point = JointTrajectoryPoint()
            point.positions = [start + blend * delta for start, delta in zip(starts, changes)]
            point.velocities = [speed * delta for delta in changes]
            point.accelerations = [acceleration * delta for delta in changes]
            point.time_from_start = Duration(
                nanoseconds=round(u * duration_s * 1_000_000_000)
            ).to_msg()
            goal.trajectory.points.append(point)

        return goal

    def move_left_gripper(self, position, duration_s=2.0, timeout_s=10.0):
        if not self.left_gripper_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("left_gripper_controller not available")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['openarm_left_finger_joint1']
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start = Duration(seconds=duration_s).to_msg()
        goal.trajectory.points = [point]

        send_future = self.left_gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout_s)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Left gripper goal rejected")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_s)
        self.get_logger().info(f"Left gripper -> {position:.3f} rad done")

    def open_left_gripper(self):
        self.move_left_gripper(GRIPPER_OPEN)

    def close_left_gripper(self, position=GRIPPER_CLOSED):
        self.move_left_gripper(position)

    @staticmethod
    def chain_moves(first, second):
        """Append `second` after `first` in one goal (for the preview); drops the duplicate joining point."""
        offset = Duration.from_msg(first.trajectory.points[-1].time_from_start).nanoseconds
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(first.trajectory.joint_names)
        goal.trajectory.points = list(first.trajectory.points)
        for p in second.trajectory.points[1:]:
            point = JointTrajectoryPoint()
            point.positions = list(p.positions)
            point.velocities = list(p.velocities)
            point.accelerations = list(p.accelerations)
            point.time_from_start = Duration(
                nanoseconds=offset + Duration.from_msg(p.time_from_start).nanoseconds
            ).to_msg()
            goal.trajectory.points.append(point)
        return goal

    @staticmethod
    def set_move_tolerances(goal, tolerance=0.03):
        goal.goal_tolerance = [
            JointTolerance(name=name, position=tolerance)
            for name in goal.trajectory.joint_names
        ]
        goal.goal_time_tolerance = Duration(seconds=2.0).to_msg()

    def execute_pair(self, left_goal, right_goal, label, record=False):
        """Schedule both arms together; wait for BOTH successful action results."""
        self.execute_goals({"left": left_goal, "right": right_goal}, label, record)

    def execute_single(self, side, goal, label):
        """Run one arm's goal and wait for its successful result."""
        self.execute_goals({side: goal}, label)

    def hold(self, seconds):
        """Wait while still processing /joint_states and action callbacks."""
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def execute_goals(self, goals, label, record=False):
        """Schedule the given arm goals together; wait for all successful results.

        Two controllers are not atomic. Rejection, timeout, interruption, or an
        arm failure triggers best-effort cancellation of all accepted goals.
        record: save /joint_states from now on, timed from the scheduled start (stopped by the caller).
        """
        all_clients = {"left": self.left_client, "right": self.right_client}
        clients = {side: all_clients[side] for side in goals}
        handles, sends, cancellations = {}, {}, []
        stopping = False
        start_delay = 3.0
        start = self.get_clock().now() + Duration(seconds=start_delay)
        if record:
            self.record_start_ns = start.nanoseconds
            self.recorded = []
            self.recording = True

        def on_response(side, future):
            if future.exception() is None:
                handle = future.result()
                if handle.accepted:
                    handles[side] = handle
                    if stopping:
                        cancellations.append(handle.cancel_goal_async())

        try:
            for side, goal in goals.items():
                goal.trajectory.header.stamp = start.to_msg()
                sends[side] = clients[side].send_goal_async(goal)
                sends[side].add_done_callback(lambda f, side=side: on_response(side, f))

            deadline = time.monotonic() + start_delay - 0.5
            while not all(f.done() for f in sends.values()):
                if not rclpy.ok() or time.monotonic() >= deadline:
                    raise RuntimeError(f"{label}: controllers did not accept in time")
                rclpy.spin_once(self, timeout_sec=0.05)
            for side, future in sends.items():
                handle = future.result()
                if not handle.accepted:
                    raise RuntimeError(f"{label}: {side} goal rejected")
                handles[side] = handle
            if self.get_clock().now().nanoseconds >= start.nanoseconds - 250_000_000:
                raise RuntimeError(f"{label}: insufficient time before scheduled start")

            self.get_logger().info(f"{label}: goals accepted; waiting for completion")
            results = {side: handle.get_result_async() for side, handle in handles.items()}
            duration = max(
                g.trajectory.points[-1].time_from_start.sec +
                g.trajectory.points[-1].time_from_start.nanosec / 1e9
                for g in goals.values()
            )
            deadline = time.monotonic() + start_delay + duration + 5.0
            pending = set(results)
            while pending:
                if not rclpy.ok() or time.monotonic() >= deadline:
                    raise RuntimeError(f"{label}: execution timed out")
                rclpy.spin_once(self, timeout_sec=0.05)
                for side in list(pending):
                    if results[side].done():
                        response = results[side].result()
                        if (response.status != GoalStatus.STATUS_SUCCEEDED or
                                response.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
                            raise RuntimeError(
                                f"{label}: {side} failed: status={response.status}, "
                                f"code={response.result.error_code}, {response.result.error_string}"
                            )
                        pending.remove(side)
                        self.get_logger().info(f"{label}: {side} completed")
        except BaseException:
            stopping = True
            for handle in handles.values():
                cancellations.append(handle.cancel_goal_async())
            # Also process late acceptances so their goals receive cancellation.
            deadline = time.monotonic() + 2.0
            while rclpy.ok() and time.monotonic() < deadline:
                if (all(f.done() for f in sends.values()) and
                        all(f.done() for f in cancellations)):
                    break
                rclpy.spin_once(self, timeout_sec=0.05)
            self.get_logger().error(f"{label}: stopped; cancellation requested for accepted goals")
            # Show which joints were off target (last /joint_states sample vs final point).
            for goal in goals.values():
                for name, target in zip(goal.trajectory.joint_names,
                                        goal.trajectory.points[-1].positions):
                    if name in self.current_positions:
                        measured = self.current_positions[name]
                        self.get_logger().error(
                            f"  {name}: target {target:+.4f}, measured {measured:+.4f}, "
                            f"error {math.degrees(measured - target):+.2f} deg"
                        )
            raise

    def wait_for_start_pose(self, goals, tolerance=0.03, timeout_s=5.0):
        """Require fresh measured angles near the CSV start before replay."""
        targets = {
            name: target
            for goal in goals
            for name, target in zip(goal.trajectory.joint_names, goal.trajectory.points[0].positions)
        }
        self.wait_for_pose(targets, "CSV start pose", tolerance, timeout_s)

    def wait_for_pose(self, targets, label, tolerance=0.03, timeout_s=5.0):
        """Require fresh measured angles within `tolerance` of `targets`; returns them."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # Discard cached feedback and require new samples for this check.
            self.current_positions.clear()
            positions = self.read_current_positions(
                timeout_s=min(1.0, max(0.01, deadline - time.monotonic()))
            )
            error = max(abs(positions[name] - target) for name, target in targets.items())
            if error <= tolerance:
                self.get_logger().info(f"Both arms at {label}; max error {error:.4f} rad")
                return positions
        raise RuntimeError(f"Arms did not reach the {label}; next motion not sent")

    def preview_approach(self, left_approach, right_approach):
        """Publish the two arms as ONE planned trajectory for RViz animation only."""
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
        from rclpy.qos import QoSProfile, DurabilityPolicy

        left = left_approach.trajectory
        right = right_approach.trajectory
        if len(left.points) != len(right.points):
            raise ValueError("Both arms must share the same approach timeline")

        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = left.joint_names + right.joint_names
        for lp, rp in zip(left.points, right.points):
            if lp.time_from_start != rp.time_from_start:
                raise ValueError("Left and right timestamps must match")
            point = JointTrajectoryPoint()
            point.positions = list(lp.positions) + list(rp.positions)
            point.velocities = list(lp.velocities) + list(rp.velocities)
            point.accelerations = list(lp.accelerations) + list(rp.accelerations)
            point.time_from_start = lp.time_from_start
            trajectory.joint_trajectory.points.append(point)

        display = DisplayTrajectory()
        display.trajectory_start.is_diff = True
        display.trajectory_start.joint_state.name = list(
            trajectory.joint_trajectory.joint_names
        )
        display.trajectory_start.joint_state.position = [0.0] * 14
        display.trajectory = [trajectory]

        self.preview_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # Retain the preview for late transient-local subscribers, and publish
        # again when a subscriber connects so volatile RViz subscribers see it too.
        self.preview_publisher.publish(display)
        previous_count = 0

        def on_subscriber_change():
            nonlocal previous_count
            count = self.preview_publisher.get_subscription_count()
            if count > previous_count:
                self.preview_publisher.publish(display)
                self.get_logger().info("RViz subscriber connected; preview published")
            previous_count = count

        self.preview_connection_timer = self.create_timer(0.5, on_subscriber_change)
        self.get_logger().info(
            "Preview ready on /display_planned_path; waiting for RViz. "
            "Use the MoveIt demo's MotionPlanning > Planned Path display. "
            "No action goals sent. Ctrl+C exits."
        )
        rclpy.spin(self)

def main():
    parser = argparse.ArgumentParser(
        description="Replay a soybean CSV through the two arm action controllers."
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to the trajectory CSV",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview-approach", action="store_true",
        help="Animate zero -> START_POSE -> CSV start in MoveIt/RViz only; sends no action goals",
    )
    mode.add_argument(
        "--execute", action="store_true",
        help="Send START_POSE move, approach, then CSV goals to the connected arm controllers",
    )
    parser.add_argument(
        "--start-pose-duration", type=float, default=10.0,
        help="Seconds for the move from the current joints to START_POSE (default: 10)",
    )
    parser.add_argument(
        "--approach-duration", type=float, default=10.0,
        help="Seconds for the move from START_POSE to the CSV start (default: 10)",
    )
    parser.add_argument(
        "--move-tolerance", type=float, default=0.03,
        help="Final position tolerance (rad) for the START_POSE and approach moves (default: 0.03)",
    )
    parser.add_argument(
        "--blend-window", type=float, default=0.3,
        help="Seconds at the CSV start retimed to start from rest; 0 = replay the CSV unchanged (default: 0.3)",
    )
    parser.add_argument(
        "--blend-duration", type=float, default=0.7,
        help="Seconds the blend window is stretched to, at most 2.5 x --blend-window (default: 0.7)",
    )
    parser.add_argument(
        "--record-dir", type=Path, default=RECORD_DIR,
        help=f"Where the run folder (commanded.csv, measured.csv) is written (default: {RECORD_DIR})",
    )
    args = parser.parse_args()
    for flag in ("start_pose_duration", "approach_duration", "move_tolerance"):
        value = getattr(args, flag)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be finite and positive")

    rclpy.init()
    node = ScoopingReplay()

    try:
        if not args.preview_approach:
            node.check_connections()
        rows = node.smooth_rows(node.load_trajectory(args.trajectory), args.blend_window, args.blend_duration)

        left_goal = node.build_goal("left", rows)
        right_goal = node.build_goal("right", rows)

        node.get_logger().info(
            f"Left goal contains {len(left_goal.trajectory.points)} points."
        )
        node.get_logger().info(
            f"Right goal contains {len(right_goal.trajectory.points)} points."
        )

        if args.preview_approach:
            zeros = dict.fromkeys(start_pose_positions(), 0.0)
            start_pose = start_pose_positions()
            left_approach = node.chain_moves(
                node.build_start_pose_move("left", zeros, args.start_pose_duration),
                node.build_approach(left_goal, start_pose, args.approach_duration),
            )
            right_approach = node.chain_moves(
                node.build_start_pose_move("right", zeros, args.start_pose_duration),
                node.build_approach(right_goal, start_pose, args.approach_duration),
            )
            node.preview_approach(left_approach, right_approach)
            return

        current_positions = node.read_current_positions()

        for side, goal in (
            ("left", left_goal),
            ("right", right_goal),
        ):
            current = [
                current_positions[name]
                for name in goal.trajectory.joint_names
            ]

            target = goal.trajectory.points[0].positions

            largest_change = max(
                abs(target_angle - current_angle)
                for target_angle, current_angle in zip(target, current)
            )

            node.get_logger().info(
                f"{side}: largest joint change to starting pose "
                f"= {math.degrees(largest_change):.1f} degrees"
            )

        left_start = [
            float(rows[0][f"openarm_left_joint{i}"])
            for i in range(1, 8)
        ]

        right_start = [
            float(rows[0][f"openarm_right_joint{i}"])
            for i in range(1, 8)
        ]

        node.get_logger().info(f"Left starting angles: {left_start}")
        node.get_logger().info(f"Right starting angles: {right_start}")

        if args.execute:
            # 1. Current joints (normally all zeros) -> START_POSE. Re-read right before
            # building the move rather than assuming the robot is at zero.
            node.current_positions.clear()
            current_positions = node.read_current_positions()
            node.get_logger().info(f"Start pose: left {START_POSE['left']}, right {START_POSE['right']}")
            left_home = node.build_start_pose_move("left", current_positions, args.start_pose_duration)
            right_home = node.build_start_pose_move("right", current_positions, args.start_pose_duration)
            for goal in (left_home, right_home):
                node.set_move_tolerances(goal, args.move_tolerance)
            node.execute_pair(left_home, right_home, "Start pose")
            current_positions = node.wait_for_pose(
                start_pose_positions(), "start pose", args.move_tolerance
            )

            # 2. Right arm only: START_POSE -> CSV first row, then pause.
            right_approach = node.build_approach(right_goal, current_positions, args.approach_duration)
            node.set_move_tolerances(right_approach, args.move_tolerance)
            node.execute_single("right", right_approach, "Right approach")
            node.wait_for_start_pose((right_goal,), args.move_tolerance)
            node.hold(2.0)

            # 3. Left gripper: quick open, hold open 3 s, quick close. The payload feed-forward is
            # dropped before the fingers open and switched on once they close on the bowl.
            node.set_payload(False)
            node.move_left_gripper(GRIPPER_OPEN, duration_s=0.5)
            node.hold(3.0)
            node.move_left_gripper(GRIPPER_CLOSED, duration_s=0.5)
            node.set_payload(True)

            # 4. Left arm only: START_POSE -> CSV first row (from fresh measurements).
            node.current_positions.clear()
            current_positions = node.read_current_positions()
            left_approach = node.build_approach(left_goal, current_positions, args.approach_duration)
            node.set_move_tolerances(left_approach, args.move_tolerance)
            node.execute_single("left", left_approach, "Left approach")

            # 5. Both arms at the CSV start -> replay together.
            node.wait_for_start_pose((left_goal, right_goal), args.move_tolerance)
            run_dir = args.record_dir / f"{args.trajectory.stem}_{datetime.now():%Y%m%d_%H%M%S}"
            run_dir.mkdir(parents=True, exist_ok=True)
            node.save_commanded(rows, run_dir)
            try:
                node.execute_pair(left_goal, right_goal, "Replay", record=True)
                node.hold(1.0)              # keep recording while the arms settle
            finally:
                node.recording = False
                node.save_recording(run_dir)   # also after a failed or interrupted replay
            node.get_logger().info("Start pose, approaches, gripper and replay completed")
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
