#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
import math
from std_msgs.msg import String
from sensor_msgs.msg import JointState

def talker():
    motors_num = 18

    pub = rospy.Publisher('/db_dynamixel_ROS_driver/hexapod_state_commands', JointState, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30) # 10hz

    joint_state = JointState()
    joint_state.name = ["id_11", "id_21", "id_31", "id_12", "id_22", "id_32", "id_13", "id_23", "id_33",   
                    "id_41", "id_51", "id_61", "id_42", "id_52", "id_62", "id_43", "id_53", "id_63"] 
    joint_state.position = [0, 0, 0, 2, 2, 2, -1, -1, -1, 
                            0, 0, 0, 2, 2, 2, -1, -1, -1]
    joint_bias = [0, 0, 0, 2, 2, 2, -1, -1, -1, 
                0, 0, 0, 2, 2, 2, -1, -1, -1]

    joint_state.velocity = [0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0]
    joint_state.effort = [200, 200, 200, 200, 200, 200, 200, 200, 200, 
                            200, 200, 200, 200, 200, 200, 200, 200, 200]

    for i in range(motors_num):
        joint_state.velocity[i] = 0
        joint_state.effort[i] = 200
    o1 = 0.9
    o2 = 0.1

    while not rospy.is_shutdown():
        o1 = math.tanh(1.4*o1 + 0.3*o2 + 0.01)
        o2 = math.tanh(1.4*o2 - 0.3*o1 + 0.01)

        o1_com = o1*0.1
        o2_com = o2*0.1
        for i in range(motors_num):
            # if i%3 == 0:
            #     joint_state.position[i] = joint_bias[i] + o2_com
            # else:
            joint_state.position[i] = joint_bias[i] + o1_com

        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(joint_state)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass