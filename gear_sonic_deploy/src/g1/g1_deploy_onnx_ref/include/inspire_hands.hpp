/**
 * @file inspire_hands.hpp
 * @brief DDS bridge for Unitree Inspire hands via rt/inspire/cmd and rt/inspire/state.
 *
 * The deploy stack uses URDF radians in policy order:
 * [thumb_yaw, thumb_pitch, index, middle, ring, pinky].
 *
 * The Inspire service uses normalized DDS commands in hardware order:
 * [pinky, ring, middle, index, thumb_bend, thumb_rotation], with 0=closed and 1=open.
 */

#ifndef INSPIRE_HANDS_HPP
#define INSPIRE_HANDS_HPP

#include <array>
#include <iostream>
#include <memory>
#include <string>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "hand_types.hpp"
#include "utils.hpp"

class InspireHands {
public:
    InspireHands() = default;

    void initialize(const std::string& ns = "inspire") {
        const std::string cmd_topic = "rt/" + ns + "/cmd";
        const std::string state_topic = "rt/" + ns + "/state";

        command_msg_.cmds().resize(kTotalMotors);
        state_msg_.states().resize(kTotalMotors);
        setAllJointsCommand(true, hand::inspireOpenPoseRad());
        setAllJointsCommand(false, hand::inspireOpenPoseRad());

        publisher_.reset(
            new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::MotorCmds_>(cmd_topic));
        subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::MotorStates_>(state_topic));

        publisher_->InitChannel();
        subscriber_->InitChannel(
            [this](const void* message) { this->onState(message); }, 1);

        std::cout << "[InspireHands] DDS topics initialized: publish " << cmd_topic
                  << ", subscribe " << state_topic << std::endl;
    }

    void writeOnce() {
        if (publisher_) {
            publisher_->Write(command_msg_);
        }
    }

    void setAllJointsCommand(bool is_left, const hand::HandJointArray& q_rad) {
        hand::HandJointArray inspire_cmd = urdfRadToInspireOrder(q_rad);
        const int offset = is_left ? static_cast<int>(hand::HAND_DOF) : 0;
        ensureCommandSize();
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            command_msg_.cmds()[offset + static_cast<int>(i)].q(
                static_cast<float>(inspire_cmd[i]));
        }
    }

    void open(bool is_left) {
        setAllJointsCommand(is_left, hand::inspireOpenPoseRad());
    }

    void close(bool is_left) {
        setAllJointsCommand(is_left, hand::inspireClosedPoseRad());
    }

    hand::HandJointArray getPosition(bool is_left) const {
        const auto data = state_buffer_.GetDataWithTime().data;
        if (!data || data->states().size() < kTotalMotors) {
            return hand::inspireOpenPoseRad();
        }

        hand::HandJointArray inspire_state{};
        const int offset = is_left ? static_cast<int>(hand::HAND_DOF) : 0;
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            inspire_state[i] = data->states()[offset + static_cast<int>(i)].q();
        }
        return inspireOrderToUrdfRad(inspire_state);
    }

    hand::HandJointArray getVelocity(bool is_left) const {
        const auto data = state_buffer_.GetDataWithTime().data;
        hand::HandJointArray dq{};
        if (!data || data->states().size() < kTotalMotors) {
            return dq;
        }

        hand::HandJointArray inspire_dq{};
        const int offset = is_left ? static_cast<int>(hand::HAND_DOF) : 0;
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            inspire_dq[i] = data->states()[offset + static_cast<int>(i)].dq();
        }

        // q_rad = lower + (1 - q_norm) * range, so dq_rad = -dq_norm * range.
        dq[0] = -inspire_dq[5] * (hand::INSPIRE_URDF_UPPER[0] - hand::INSPIRE_URDF_LOWER[0]);
        dq[1] = -inspire_dq[4] * (hand::INSPIRE_URDF_UPPER[1] - hand::INSPIRE_URDF_LOWER[1]);
        dq[2] = -inspire_dq[3] * (hand::INSPIRE_URDF_UPPER[2] - hand::INSPIRE_URDF_LOWER[2]);
        dq[3] = -inspire_dq[2] * (hand::INSPIRE_URDF_UPPER[3] - hand::INSPIRE_URDF_LOWER[3]);
        dq[4] = -inspire_dq[1] * (hand::INSPIRE_URDF_UPPER[4] - hand::INSPIRE_URDF_LOWER[4]);
        dq[5] = -inspire_dq[0] * (hand::INSPIRE_URDF_UPPER[5] - hand::INSPIRE_URDF_LOWER[5]);
        return dq;
    }

    bool hasState() const {
        return state_buffer_.GetDataWithTime().HasData();
    }

    static hand::HandJointArray urdfRadToInspireOrder(const hand::HandJointArray& q_rad) {
        hand::HandJointArray cmd_dataset_order{};
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            cmd_dataset_order[i] = hand::radToInspireCommand(q_rad[i], i);
        }

        return {
            cmd_dataset_order[5],  // pinky
            cmd_dataset_order[4],  // ring
            cmd_dataset_order[3],  // middle
            cmd_dataset_order[2],  // index
            cmd_dataset_order[1],  // thumb_bend
            cmd_dataset_order[0],  // thumb_rotation
        };
    }

    static hand::HandJointArray inspireOrderToUrdfRad(const hand::HandJointArray& inspire_q) {
        hand::HandJointArray dataset_order = {
            inspire_q[5],  // thumb_yaw
            inspire_q[4],  // thumb_pitch
            inspire_q[3],  // index
            inspire_q[2],  // middle
            inspire_q[1],  // ring
            inspire_q[0],  // pinky
        };

        hand::HandJointArray q_rad{};
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            q_rad[i] = hand::inspireCommandToRad(dataset_order[i], i);
        }
        return q_rad;
    }

private:
    static constexpr std::size_t kTotalMotors = hand::HAND_DOF * 2;

    void ensureCommandSize() {
        if (command_msg_.cmds().size() != kTotalMotors) {
            command_msg_.cmds().resize(kTotalMotors);
        }
    }

    void onState(const void* message) {
        const auto* incoming =
            static_cast<const unitree_go::msg::dds_::MotorStates_*>(message);
        if (!incoming || incoming->states().size() < kTotalMotors) {
            return;
        }
        state_buffer_.SetData(*incoming);
    }

    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::MotorCmds_> publisher_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::MotorStates_> subscriber_;
    unitree_go::msg::dds_::MotorCmds_ command_msg_;
    unitree_go::msg::dds_::MotorStates_ state_msg_;
    DataBuffer<unitree_go::msg::dds_::MotorStates_> state_buffer_;
};

#endif  // INSPIRE_HANDS_HPP
