# Setup matplotlib animation
import matplotlib.pyplot as plt


from utils import *
from agents import *
from search import *
from problem import *
from fire_problem import *
from testing_utils import *



#%%

# problem = get_problem("just_wait")
# agent = FireMDPDeterminizedAStarAgent(problem)
# trajectory = run_agent_on_problem(problem, agent)
# anim = animate_trajectory(problem, trajectory)
# from matplotlib.animation import FFMpegWriter
# anim.save("mp4/videos/just_wait_.mp4", writer=FFMpegWriter(fps=5))

#%%

# problem = get_problem("the_choice")
# agent = FireMDPDeterminizedAStarAgent(problem)
# trajectory = run_agent_on_problem(problem, agent)
# anim = animate_trajectory(problem, trajectory)
# from matplotlib.animation import FFMpegWriter
# anim.save("mp4/videos/the_choice_determinized_.mp4", writer=FFMpegWriter(fps=5))

# # #%%

problem = get_problem("the_choice")
agent = MCTSAgent(problem)
trajectory = run_agent_on_problem(problem, agent)
anim = animate_trajectory(problem, trajectory)
from matplotlib.animation import FFMpegWriter
anim.save("mp4/videos/the_choice_mcts__.mp4", writer=FFMpegWriter(fps=5))
