import torch

def get_num_legjoints(robot):
    try:
        if robot == 'Dbalpha':
            num_legs = 6
            num_joints = 3
            motor_mapping = torch.tensor([3, 6, 12, 15, 0, 9,
                                        4, 7, 13, 16, 1, 10, 
                                        5, 8, 14, 17, 2, 11])
        elif robot == 'Ant' or robot == 'Ant_test':
            num_legs = 4
            num_joints = 2
            motor_mapping = torch.tensor([0, 2, 6, 4, 1, 3, 7, 5])
    except:
        print("error get_num_legjoints function, please recheck the name of the robot")
    return num_legs, num_joints, motor_mapping