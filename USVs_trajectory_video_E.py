import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
from math import radians, sin, cos
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib
import yaml
from PIL import Image
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
from collections import defaultdict
import math
from matplotlib import font_manager

font_path = 'simsun.ttc'  # Windows 系统下的微软雅黑字体路径
prop = font_manager.FontProperties(fname=font_path)
#plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 手动指定 FFmpeg 路径（根据实际安装路径修改）
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\LENOVO\Downloads\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe'
time_interval = 20/60
def generate_v_h_data(energy=True,same_st=True,model_name='MLP'):
    """加载三种方案的轨迹数据"""
    solutions = []
    #posi = (4.0, 0.0)  # 固定起始位置
    #scale = 100

    # 获取四个无人船的轨迹
    # 获取四个无人船的轨迹
    if energy:
        if same_st:
            schedule_file = 'results/trajectory/output_energy_{}_SP.yaml'.format(model_name)
        else:
            schedule_file = 'results/trajectory/output_energy_{}_DP.yaml'.format(model_name)
    else:
        if same_st:
            schedule_file = 'results/trajectory/output_no_energy_{}_SP.yaml'.format(model_name)
        else:
            schedule_file = 'results/trajectory/output_no_energy_{}_DP.yaml'.format(model_name)

    # 颜色和标签配置
    configs = [
        {'color':  '#1f77b4', 'label': '无人船1'},  # 深蓝
        {'color': '#ff7f0e', 'label': '无人船2'},  # 红色
        {'color': '#2ca02c', 'label': '无人船3'},  # 绿色
        {'color': '#d62728', 'label': '无人船4'},  # 绿色
        {'color': 'black', 'label': '动态障碍物'},  # 黑色
        {'color': 'black', 'label': '动态障碍物'},  # 黑色
        {'color': 'black', 'label': '动态障碍物'},  # 黑色
        {'color': 'black', 'label': '动态障碍物'}
    ]
    with open(schedule_file) as states_file:
        schedule = yaml.load(states_file, Loader=yaml.FullLoader)

    for i,(name, positions) in enumerate(schedule["schedule"].items()):
        solution = defaultdict(list)
        if name in ['obs0', 'obs1', 'obs2', 'obs3']:
            for pos in positions:
                solution['x'].append(pos['x'])
                solution['y'].append(pos['y'])
            solutions.append({
                'x': np.array(solution['x']),
                'y': np.array(solution['y']),
                **configs[i]
            })
        else:
            for pos in positions:
                solution['v'].append(pos['v'])
                solution['heading'].append(pos['heading'])
                solution['load'].append(pos['load'])
                solution['E'].append(pos['E'])
                solution['ARC'].append(pos['ARC'])

            solutions.append({
                'v': np.array(solution['v']),
                'heading': np.array(solution['heading']),
                'load': np.array(solution['load']),
                'E': np.array(solution['E']),
                'ARC': np.array(solution['ARC']),
                **configs[i]
            })

    return solutions


def calculate_trajectory(speed, heading,label, dt= 20/60,same_st = True):
    """带边界限制的轨迹计算"""
    if same_st:
        state = np.load('results/trajectory/start_ST.npy')
    else:
        state = np.load('results/trajectory/start_DT.npy')
    #print(label)
    #print(state)
    if label== '无人船1':
        start = state[0:2]
        goal = state[2:4]
    elif label == '无人船2':
        start = state[4:6]
        goal = state[6:8]
    elif label == '无人船3':
        start = state[8:10]
        goal = state[10:12]
    elif label == '无人船4':
        start = state[12:14]
        goal = state[14:16]

    x = np.zeros(len(speed)+1)
    y = np.zeros(len(speed)+1)
    x[0] = start[0]
    y[0] = start[1]
    #angles = np.radians(90 - heading)

    for i in range(0, len(speed)):
        dx = speed[i] * dt *math.cos(heading[i])
        dy = speed[i] * dt *math.sin(heading[i])
        x[i+1] = x[i] + dx
        y[i+1] = y[i] + dy

        # 边界约束
        #x[i] = np.clip(x[i], *x_limits)
        #y[i] = np.clip(y[i], *y_limits)

    return x, y,speed,heading
scale = 100
# 视频生成函数
def generate_video(trajectories, max_len=96,output_file='trajectory.mp4', fps=30,x_limits=(-500, 500),    # 新增x边界参数
                  y_limits=(-500, 500)):
    """生成航行轨迹视频
    参数：
        x, y: 坐标数组
        speed: 速度数组
        heading: 航向数组
        output_file: 输出文件名
        fps: 视频帧率
    """
    map_file = "env/map_8by8_obst12_agents4_ex58.yaml"
    with open(map_file) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    #state = np.load('result_models/CSV/start_{}_{}.npy'.format(4.0, 0.0))
    # 绘制起始点和目标点
    # start = state[:2]*scale
    # goal = state[2:]*scale
    # plt.gca().add_patch(plt.Circle((goal[0] * scale, goal[1] * scale), 10, color='#FFA500', label='目标区域'))
    # plt.gca().add_patch(plt.Circle((start[0] * scale, start[1] * scale), 5, color='#32CD32', label='起点'))
    obstacles = map["map"]["obstacles_static"]
    # print(obstacles)
    # obstacles_x_list = []
    # obstacles_y_list = []
    for i, o in enumerate(obstacles):
        x_, y_ = o[0] * scale, o[1] * scale
        # Rectangle((x - 0.5, y - 0.5), 0.1, 0.1, facecolor='red', edgecolor='red')
        if i == 0:
            circle = plt.Circle((x_, y_), 20, color='r',label='危险区域')

        else:
            circle = plt.Circle((x_, y_), 20, color='r')
        # plt.legend()
        plt.gca().add_patch(circle)
        plt.axis('scaled')

    colors = {-0.1: 'r', -0.2: 'b', 0: 'c', 0.1: 'yellow', 0.2: 'm', 0.3: 'g', 0.4: 'k'}
    with open('env/sea_state.yaml', 'r') as sea_file:
        try:
            sea_states = yaml.load(sea_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    ARC_EXIT = set()
    for name, pos in sea_states.items():
        # print(pos)
        x_min = pos['x_min']
        x_max = pos['x_max']
        y_min = pos['y_min']
        y_max = pos['y_max']
        ARC = pos['ARC']
        if ARC not in ARC_EXIT:
            # print('legend')
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                      (y_max - y_min) * scale,
                                      # fill=False,
                                      alpha=0.3,
                                      facecolor=colors[ARC], label='${}$'.format(ARC))
        else:
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                      (y_max - y_min) * scale,
                                      # fill=False,
                                      alpha=0.3,
                                      facecolor=colors[ARC])
        if len(ARC_EXIT) < len(colors):
            ARC_EXIT.add(ARC)
        plt.gca().add_patch(rect)

    # 移除所有边距
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # 设置背景颜色（RGBA格式）
    fig.patch.set_facecolor('#F0F4F8')  # 浅蓝灰色背景
    ax.set_facecolor('white')  # 坐标区白色

    # 计算最大轨迹长度
    max_length = max_len#max(len(t['x']) for t in trajectories[:4])
    #print(max_length)
    # 填充较短轨迹的末端数据
    for traj in trajectories:
        #print(traj['label'])
        if len(traj['x']) < max_length:
            last_x = traj['x'][-1]
            last_y = traj['y'][-1]
            traj['x'] = np.concatenate([traj['x'], np.full(max_length - len(traj['x']), last_x)])
            traj['y'] = np.concatenate([traj['y'], np.full(max_length - len(traj['y']), last_y)])
        if len(traj['x']) > max_length:
            traj['x'] = traj['x'][:max_length]
            traj['y'] = traj['y'][:max_length]

    # 设置固定坐标范围
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    # 初始化轨迹元素
    lines = []
    points = []
    for traj in trajectories:
        if traj['label'] != '动态障碍物':
            line, = ax.plot([], [], lw=2, color=traj['color'], label=traj['label'])
            point = ax.scatter([], [], s=20, color=traj['color'], zorder=4)
        else:
            line, = ax.plot([], [], lw=1, color=traj['color'])
            point = ax.scatter([], [], s=20, color=traj['color'], zorder=4)
        lines.append(line)
        points.append(point)

    # 时间显示
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,fontproperties=prop,
                        fontsize=12, color='black', fontweight='bold')
    # info_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
    #                     color='#555555', fontsize=10)

    # 设置坐标轴
    #ax.set_xlabel('X (nm)')
    #ax.set_ylabel('Y (nm)')
    #ax.set_title('Ship Trajectory')
    ax.grid(True, alpha=0.3)


    # 添加动态标签对象（优化样式）
    label_texts = []
    #print(trajectories[4:])
    for i, traj in enumerate(trajectories[:4]):
        text = ax.text(
            0, 0, '',
            color='white',
            fontsize=10,  # 减小字号
            fontproperties=prop,
            fontweight='bold',
            bbox=dict(
                facecolor=traj['color'],
                alpha=0.8,
                edgecolor='black',
                boxstyle='round,pad=0.3',  # 紧凑内边距
                linewidth=1
            ),
            verticalalignment='bottom',
            horizontalalignment='left',
            linespacing=1.25,  # 紧凑行距
            wrap=True
        )
        # 添加固定宽度约束
        text._get_wrap_line_width = lambda: 150  # 单位：像素
        label_texts.append(text)

    # 动画更新函数
    # 添加连接线存储
    connection_lines = [ax.plot([], [], color=traj['color'], alpha=0.3, linestyle='--')[0]
                        for traj in trajectories[:4]]

    # 动画更新函数（优化版）
    def update(frame):
        updates = []
        actual_frame = min(frame, max_length - 1)

        # 存储当前所有标签位置
        label_positions = []

        # 第一遍：计算初始位置
        temp_positions = []
        for i, traj in enumerate(trajectories):
            current_frame = min(actual_frame, len(traj['x']) - 1)
            current_x = traj['x'][current_frame] * scale
            current_y = traj['y'][current_frame] * scale

            # 初始偏移方向（按索引分配角度）
            angle = 45 * (i % 8)  # 8个方向循环
            dx = 35 * math.cos(math.radians(angle))
            dy = 35 * math.sin(math.radians(angle))
            temp_positions.append((
                current_x + dx,
                current_y + dy,
                current_x,  # 原始坐标用于连接线
                current_y
            ))

        # 第二遍：调整重叠（修复版）
        final_positions = []
        for i, pos in enumerate(temp_positions):
            new_x, new_y = pos[0], pos[1]
            base_x, base_y = pos[2], pos[3]

            # 添加空列表保护
            if final_positions:
                # 计算与所有现有标签的最小距离
                min_distance = min(
                    math.hypot(new_x - ex, new_y - ey)
                    for (ex, ey) in final_positions
                )

                # 动态调整直到满足最小间距
                adjust_step = 0
                while min_distance < 100 and adjust_step < 10:
                    # 向右上方移动
                    new_x += 30 * (adjust_step + 1)
                    new_y += 15 * (adjust_step + 1)

                    # 更新最小距离
                    min_distance = min(
                        math.hypot(new_x - ex, new_y - ey)
                        for (ex, ey) in final_positions
                    )
                    adjust_step += 1

            # 添加最终位置
            final_positions.append((
                max(x_limits[0] + 20, min(x_limits[1] - 20, new_x)),
                max(y_limits[0] + 20, min(y_limits[1] - 20, new_y))
            ))

        # 第三遍：更新显示
        for i, traj in enumerate(trajectories):
            current_frame = min(actual_frame, len(traj['x']) - 1)
            current_x = traj['x'][current_frame] * scale
            current_y = traj['y'][current_frame] * scale

            # 获取调整后的位置
            label_x, label_y = final_positions[i][0], final_positions[i][1]

            # 边界约束
            label_x = max(x_limits[0] + 50, min(x_limits[1] - 50, label_x))
            label_y = max(y_limits[0] + 50, min(y_limits[1] - 50, label_y))


            if i < 4:
                # 更新连接线
                connection_lines[i].set_data([current_x, label_x], [current_y, label_y])
                # 更新标签位置
                label_texts[i].set_position((label_x, label_y))

                # 更新数据
                v = traj['v'][current_frame] * scale if current_frame < len(traj['v']) else traj['v'][-1] * scale
                heading = traj['heading'][current_frame] * 180 / np.pi if current_frame < len(traj['heading']) else \
                traj['heading'][-1] * 180 / np.pi
                E = traj['E'][current_frame] if current_frame < len(traj['E']) else traj['E'][-1]
                load = traj['load'][current_frame] if current_frame < len(traj['load']) else traj['E'][-1]
                ARC = traj['ARC'][current_frame] if current_frame < len(traj['ARC']) else traj['ARC'][-1]

                # label_texts[i].set_text(
                #     f"{traj['label']}\n"
                #  f"航速: {v:.2f} kn\n"
                #     f"航向: {heading:.1f}°\n"
                #     f"能耗: {heading:.1f}°\n"
                #      )
                label_texts[i].set_text(
                    f"{traj['label']}\n"
                    f"航速:{v:.2f}kn 航向:{heading:.2f}°\n"  # 合并为单行显示
                    f"附加阻力系数:{ARC:.1f} 负载:{load:.2f}kW"  #
                )
                # 在设置文本后添加自适应调整
                #text_bbox = label_texts[i].get_bbox_patch()
                #text_bbox.set_boxstyle(f"round,pad=0.2,rounding_size={0.15 * (1 + frame / max_length)}")  # 动态圆角
                updates.extend([lines[i], points[i], label_texts[i], connection_lines[i]])

            # 更新轨迹元素
            lines[i].set_data(
                traj['x'][:current_frame + 1] * scale,
                traj['y'][:current_frame + 1] * scale
            )
            points[i].set_offsets([[current_x, current_y]])



        # 更新时间显示
        minutes = actual_frame * time_interval
        time_text.set_text(f"时间: {minutes:.2f}小时")
        updates.append(time_text)

        return updates

    fig.legend(ncol=6, bbox_to_anchor=(0.78, 0.94),prop=prop)
    # 创建动画
    #min_length = min(len(t['x']) for t in trajectories)
    ani = FuncAnimation(fig, update, frames=max_length,
                        interval=1000 // fps, blit=True)

    # 视频编码设置
    writer = FFMpegWriter(
        fps=fps,
        metadata={'title': '航行轨迹比较'},
        extra_args=['-preset', 'slow', '-crf', '18']
    )

    # 保存视频
    ani.save(output_file, writer=writer, dpi=150)
    print(f'视频已保存至: {output_file}')
    plt.close()



# 主程序
if __name__ == "__main__":
    energy= True
    same_st = False
    model_name ='MHA'
    max_len = 92
    if energy:
        if same_st:
            out_file = 'output_energy_{}_SP_E.mp4'.format(model_name)
        else:
            out_file = 'output_energy_{}_DP_E.mp4'.format(model_name)
    else:
        if same_st:
            out_file = 'output_no_energy_{}_SP_E.mp4'.format(model_name)
        else:
            out_file = 'output_no_energy_{}_DP_E.mp4'.format(model_name)

    solutions = generate_v_h_data(energy=energy, same_st=same_st, model_name=model_name)
    # 计算各方案轨迹
    trajectories = []
    for solution in solutions:
        if solution['label']  != '动态障碍物':
            x, y, v, headig = calculate_trajectory(
                solution['v'],
                solution['heading'],
                solution['label'],
                same_st=same_st )
            # print(x,y)
            trajectories.append({
                'x': x,
                'y': y,
                'v': v,
                'heading': headig,
                'load': solution['load'],
                'E': solution['E'],
                'ARC': solution['ARC'],
                'color': solution['color'],
                'label': solution['label']
            })
        else:
            #print(solution)
            trajectories.append({
                'x': solution['x'],
                'y': solution['y'],
                'v': 0,
                'heading': 0,
                'color': solution['color'],
                'label': solution['label']
            })

    # 生成对比视频
    generate_video(
        trajectories,
        output_file=out_file,
        max_len=max_len,
        fps=4,
        x_limits=(0, 800),
        y_limits=(0, 800)
    )
