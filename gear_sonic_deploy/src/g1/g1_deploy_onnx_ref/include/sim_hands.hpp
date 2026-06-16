/**
 * @file sim_hands.hpp
 * @brief MuJoCo Inspire-hand DDS bridge using URDF-radian values.
 *
 * DDS uses the Inspire topic/type/order:
 * [pinky, ring, middle, index, thumb_bend, thumb_rotation],
 * right hand first, then left hand. Unlike the real hardware bridge, values
 * remain radians for MuJoCo and are not normalized to [0,1].
 */

#ifndef SIM_HANDS_HPP
#define SIM_HANDS_HPP

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "hand_types.hpp"
#include "utils.hpp"

class SimHands {
public:
    SimHands() = default;

    void initialize() {
        command_msg_.cmds().resize(kTotalMotors);
        setAllJointsCommand(true, hand::inspireOpenPoseRad());
        setAllJointsCommand(false, hand::inspireOpenPoseRad());

        publisher_.reset(
            new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::MotorCmds_>(
                "rt/inspire/cmd"));
        subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::MotorStates_>(
                "rt/inspire/state"));

        publisher_->InitChannel();
        subscriber_->InitChannel(
            [this](const void* message) { this->onState(message); }, 1);

        std::cout << "[SimHands] DDS topics initialized: rt/inspire/* with URDF-radian values"
                  << std::endl;
    }

    void writeOnce() {
        if (publisher_) {
            publisher_->Write(command_msg_);
        }
    }

    void setAllJointsCommand(bool is_left, const hand::HandJointArray& q_rad) {
        ensureCommandSize();
        const int offset = is_left ? static_cast<int>(hand::HAND_DOF) : 0;
        const hand::HandJointArray inspire_order = urdfRadToInspireRadOrder(q_rad);
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            auto& cmd = command_msg_.cmds()[offset + static_cast<int>(i)];
            cmd.q(static_cast<float>(inspire_order[i]));
            cmd.dq(0.0F);
            cmd.kp(kSimHandKp);
            cmd.kd(kSimHandKd);
            cmd.tau(0.0F);
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

        hand::HandJointArray inspire_q{};
        const int offset = is_left ? static_cast<int>(hand::HAND_DOF) : 0;
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            inspire_q[i] = data->states()[offset + static_cast<int>(i)].q();
        }
        return inspireRadOrderToUrdfRad(inspire_q);
    }

    hand::HandJointArray getVelocity(bool is_left) const {
        const auto data = state_buffer_.GetDataWithTime().data;
        if (!data || data->states().size() < kTotalMotors) {
            return {};
        }

        hand::HandJointArray inspire_dq{};
        const int offset = is_left ? static_cast<int>(hand::HAND_DOF) : 0;
        for (std::size_t i = 0; i < hand::HAND_DOF; ++i) {
            inspire_dq[i] = data->states()[offset + static_cast<int>(i)].dq();
        }
        return inspireRadOrderToUrdfRad(inspire_dq);
    }

private:
    static constexpr std::size_t kTotalMotors = hand::HAND_DOF * 2;
    static constexpr float kSimHandKp = 1.5F;
    static constexpr float kSimHandKd = 0.1F;

    void ensureCommandSize() {
        if (command_msg_.cmds().size() != kTotalMotors) {
            command_msg_.cmds().resize(kTotalMotors);
        }
    }

    static hand::HandJointArray urdfRadToInspireRadOrder(
        const hand::HandJointArray& q_rad) {
        return {
            q_rad[5],
            q_rad[4],
            q_rad[3],
            q_rad[2],
            q_rad[1],
            q_rad[0],
        };
    }

    static hand::HandJointArray inspireRadOrderToUrdfRad(
        const hand::HandJointArray& inspire_q) {
        return {
            inspire_q[5],
            inspire_q[4],
            inspire_q[3],
            inspire_q[2],
            inspire_q[1],
            inspire_q[0],
        };
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
    DataBuffer<unitree_go::msg::dds_::MotorStates_> state_buffer_;
};

#endif  // SIM_HANDS_HPP
