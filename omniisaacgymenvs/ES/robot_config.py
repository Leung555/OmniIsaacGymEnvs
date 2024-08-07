import torch

def get_num_legjoints(robot):
    # Ant robot variation
    Ant_env = ['Ant', 'Ant_test', 'Ant_sim_rew']
    try:
        if robot == 'Dbalpha':
            num_legs = 6
            num_joints = 3
            motor_mapping = torch.tensor([3, 6, 12, 15, 0, 9,
                                        4, 7, 13, 16, 1, 10, 
                                        5, 8, 14, 17, 2, 11])
        if robot == 'Slalom':
            num_legs = 4
            num_joints = 6
            motor_mapping = torch.tensor([0, 12, 6, 18, 1, 13, 7, 19,  
                                          2, 14, 8, 20, 3, 15, 9, 21, 
                                          4, 16, 10, 22, 5, 17, 11, 23]
)
        elif robot in Ant_env:
            num_legs = 4
            num_joints = 2
            motor_mapping = torch.tensor([0, 2, 6, 4, 1, 3, 7, 5])
    except:
        print("error get_num_legjoints function, please recheck the name of the robot")
    return num_legs, num_joints, motor_mapping