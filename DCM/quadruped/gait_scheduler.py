from copy import deepcopy
import numpy as np
import time
import cv2

class GaitScheduler:
    def __init__(self, args):
        self.num_leg = args['num_leg']
        self.init_gait_period = args['gait_period']
        self.is_visualize = args['is_visualize']
        self.visualize_period = 5
        self.init_gait_list = np.zeros((self.num_leg, 2)) # FR, FL, RR, RL | 0 <= offset < 1, 0 <= duration < 1


    def reset(self):
        self.gait_list = deepcopy(self.init_gait_list)
        self.next_gait_list = deepcopy(self.init_gait_list)
        self.gait_period_list = [self.init_gait_period, self.init_gait_period]
        self.next_gait_period = self.init_gait_period
        self.gait_freq = 1/self.gait_period_list[0]
        self.swing_time_list = []
        self.gait_phase = 0.0
        self.update()

        self.swing_elapsed_time_list = np.zeros(self.num_leg)
        self.pre_contact_list = np.ones(self.num_leg)

        self.visualize_cnt = 0

    def get_contact_list(self, time_step, time_horizon):
        contact_list = np.ones((time_horizon, self.num_leg))
        step_info_list = []
        for leg_idx in range(self.num_leg):
            step_info_list.append([False, []])

            swing_time_list = self.swing_time_list[leg_idx]
            if len(swing_time_list) == 0:
                continue

            swing_cnt = 0
            for swing_time in swing_time_list:
                start_phase, end_phase, swing_period = swing_time
                if end_phase < self.gait_phase:
                    continue

                if start_phase < 1.0:
                    t1 = start_phase*self.gait_period_list[0]
                else:
                    t1 = self.gait_period_list[0] + (start_phase - 1.0)*self.gait_period_list[1]
                if self.gait_phase < 1.0:
                    curr_t = self.gait_phase*self.gait_period_list[0]
                else:
                    curr_t = self.gait_period_list[0] + (self.gait_phase - 1.0)*self.gait_period_list[1]
                t2 = t1 + swing_period
                idx1 = int(round(max(t1 - curr_t, 0.0)/time_step))
                idx2 = int(round(max(t2 - curr_t, 0.0)/time_step))
                if idx1 >= time_horizon:
                    break
                idx2 = min(time_horizon, idx2)
                if idx1 == idx2:
                    continue

                contact_list[idx1:idx2, leg_idx] = 0.0

                stand_period = (1.0 - self.gait_list[leg_idx][1])*self.gait_period_list[1] # duration * gait period
                if swing_cnt == 0 and start_phase < self.gait_phase:
                    step_info_list[-1][0] = True
                step_info_list[-1][1].append([stand_period, idx1, idx2])
                swing_cnt += 1

        return contact_list, step_info_list
        
    def update(self):
            self.gait_period_list = [self.gait_period_list[1], self.next_gait_period]
            self.gait_freq = 1/self.gait_period_list[0]
            self.swing_time_list = []
            for leg_idx in range(self.num_leg):
                o_k, d_k = self.gait_list[leg_idx]
                temp_swing_time_list = self.get_swing_time_list(o_k, d_k, self.gait_period_list[0], is_next=False)
                o_k2, d_k2 = self.next_gait_list[leg_idx]
                temp_swing_time_list2 = self.get_swing_time_list(o_k2, d_k2, self.gait_period_list[1], is_next=True)
                if len(temp_swing_time_list) != 0 and len(temp_swing_time_list2) != 0 and \
                                    temp_swing_time_list[-1][1] == temp_swing_time_list2[0][0]:
                    temp_swing_time_list2[0][0] = temp_swing_time_list[-1][0]
                    temp_swing_time_list2[0][2] += temp_swing_time_list[-1][2]
                    del(temp_swing_time_list[-1])
                self.swing_time_list.append(temp_swing_time_list + temp_swing_time_list2)
            self.gait_list = deepcopy(self.next_gait_list)

    def step(self, time_step):
        if self.gait_phase >= 1.0:
            self.gait_phase -= 1.0
            self.update()

        contact_list = np.ones(self.num_leg)
        gait_info_list = []
        for leg_idx in range(self.num_leg):
            temp_gait_info_list = []
            swing_time_list = self.swing_time_list[leg_idx]
            for swing_time in swing_time_list:
                if self.gait_phase >= swing_time[0] and self.gait_phase <= swing_time[1]:
                    contact_list[leg_idx] = 0.0
                    if self.pre_contact_list[leg_idx] == 1.0:
                        self.swing_elapsed_time_list[leg_idx] = 0.0
                    swing_period = swing_time[2]
                    stand_period = (1.0 - self.gait_list[leg_idx][1])*self.gait_period_list[1] # stand duration * gait period
                    temp_gait_info_list.append(swing_period)
                    temp_gait_info_list.append(self.swing_elapsed_time_list[leg_idx])
                    self.swing_elapsed_time_list[leg_idx] += time_step
                    break
                elif self.gait_phase < swing_time[0]:
                    break
            gait_info_list.append(temp_gait_info_list)
        self.pre_contact_list = deepcopy(contact_list)

        if self.is_visualize:
            self.visualize_cnt += 1
            if self.visualize_cnt == self.visualize_period:
                self.visualize_cnt = 0
                self.visualize()

        self.gait_phase += time_step*self.gait_freq
        return contact_list, gait_info_list
        
    def set_gait_list(self, gait_list, gait_period=None):
        self.next_gait_list = deepcopy(gait_list)
        if gait_period != None:
            self.next_gait_period = gait_period

    def close(self):
        if self.is_visualize:
            cv2.destroyWindow('gait')

    def visualize(self):
        img = np.zeros((200, 400, 3), np.uint8)
        for leg_idx in range(self.num_leg):
            temp_swing_time_list = self.swing_time_list[leg_idx]
            for swing_time in temp_swing_time_list:
                if swing_time[1] < self.gait_phase or swing_time[0] - self.gait_phase > 1.0:
                    continue
                start_time = max(0.0, swing_time[0] - self.gait_phase)
                end_time = min(1.0, swing_time[1] - self.gait_phase)
                pt1 = (int(start_time*img.shape[1]), leg_idx*int(img.shape[0]/self.num_leg))
                pt2 = (int(end_time*img.shape[1]), (leg_idx + 1)*int(img.shape[0]/self.num_leg))
                img = cv2.rectangle(img, pt1, pt2, (255,255,255), -1)
        for leg_idx in range(self.num_leg - 1):
            pt1 = (0, (leg_idx + 1)*int(img.shape[0]/self.num_leg))
            pt2 = (img.shape[1], (leg_idx + 1)*int(img.shape[0]/self.num_leg))
            img = cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        cv2.imshow('gait', img)
        cv2.waitKey(1)

    def get_swing_time_list(self, offset, duration, period, is_next=False):
        temp_gait_time_list = []
        o_k, d_k = offset, duration
        if is_next:
            if d_k == 0:
                pass
            elif d_k == 1:
                temp_gait_time_list.append([1.0, 2.0, period])
            elif o_k + d_k > 1:
                temp_gait_time_list.append([1.0, o_k + d_k, (o_k + d_k - 1.0)*period])
                temp_gait_time_list.append([1.0 + o_k, 2.0, (1.0 - o_k)*period])
            elif o_k + d_k == 1:
                temp_gait_time_list.append([1.0 + o_k, 2.0, (1.0 - o_k)*period])
            else:
                temp_gait_time_list.append([1.0 + o_k, o_k + d_k + 1.0, d_k*period])
        else:
            if d_k == 0:
                pass
            elif d_k == 1:
                temp_gait_time_list.append([0.0, 1.0, period])
            elif o_k + d_k > 1:
                temp_gait_time_list.append([0.0, o_k + d_k - 1, (o_k + d_k - 1.0)*period])
                temp_gait_time_list.append([o_k, 1.0, (1.0 - o_k)*period])
            elif o_k + d_k == 1:
                temp_gait_time_list.append([o_k, 1.0, (1.0 - o_k)*period])
            else:
                temp_gait_time_list.append([o_k, o_k + d_k, d_k*period])
        return temp_gait_time_list


if __name__ == "__main__":
    num_leg = 4
    time_step = 1e-3
    global_time = 0.0

    gs_args = dict()
    gs_args['num_leg'] = num_leg
    gs_args['gait_period'] = 1.0
    gs_args['is_visualize'] = True #False
    gait_scheduler = GaitScheduler(gs_args)

    gait_scheduler.reset()

    while global_time < 10.0:
        
        if global_time >= 5.0:
            gait_list = np.zeros((num_leg, 2)) # FR, FL, RR, RL
            gait_list[0,0] = 0.5 # offset
            gait_list[0,1] = 0.5 # duration
            gait_list[2,0] = 0.0 # offset
            gait_list[2,1] = 0.5 # duration
            gait_period = 2.0
            gait_scheduler.set_gait_list(gait_list, gait_period)
        elif global_time >= 1.0:
            gait_list = np.zeros((num_leg, 2)) # FR, FL, RR, RL
            gait_list[0,0] = 0.5 # offset
            gait_list[0,1] = 0.5 # duration
            gait_period = 1.0
            gait_scheduler.set_gait_list(gait_list, gait_period)

        contact_list, step_info_list = gait_scheduler.get_contact_list(0.05, 10)
        print(step_info_list)

        contact_list, gait_info_list = gait_scheduler.step(time_step)

        global_time += time_step
        
    gait_scheduler.close()
