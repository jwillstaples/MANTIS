import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def find_floats(line: str) -> tuple:
    colon_1 = line.find(":")
    comma_1 = line.find(",")
    float_1 = np.float64(line[colon_1 + 1 : comma_1].strip())

    colon_2 = line[comma_1 + 1 :].find(":") + comma_1 + 1
    comma_2 = line[comma_1 + 1 :].find(",") + comma_1 + 1
    float_2 = np.float64(line[colon_2 + 1 : comma_2].strip())

    if line[0] == "T":
        colon_3 = line[comma_2 + 1 :].find("=") + comma_2 + 1
        comma_3 = -2
        float_3 = np.float64(line[colon_3 + 1 : comma_3].strip())
    else:
        colon_3 = line[comma_2 + 1 :].find(":") + comma_2 + 1
        comma_3 = -1
        float_3 = np.float64(line[colon_3 + 1 : comma_3].strip())

    return float_1, float_2, float_3


class Run:

    def __init__(
        self,
        num_games: int,
        num_cores: int,
        total_times: np.ndarray,
        forward_times: np.ndarray,
    ):
        self.num_games = num_games
        self.num_cores = num_cores
        self.total_times = total_times
        self.forward_times = forward_times

    @classmethod
    def from_block(cls, block: list[str]):

        header = block[0]
        num_games, generated_per_core, num_cores = find_floats(header)

        total_times = []
        forward_times = []
        for i in range(1, len(block)):
            total_time, forward_time, _ = find_floats(block[i])
            total_times.append(total_time)
            forward_times.append(forward_time)

        return cls(num_games, num_cores, np.array(total_times), np.array(forward_times))


def graph_cores(data: np.ndarray):

    ### num_core dependency
    f, axs = plt.subplots(2, 1, dpi=600, figsize=(6, 4.5))
    for run in data:
        if run.num_games == 1000:
            axs[0].scatter(run.num_cores, np.mean(run.total_times), color="firebrick")
            axs[0].scatter(run.num_cores, np.max(run.total_times), color="blue")
            axs[1].scatter(
                run.num_cores,
                np.mean(100 * run.forward_times / run.total_times),
                color="forestgreen",
            )

        f.suptitle("Dependency on Number of Cores")

    axs[1].set_xlabel("Number of Cores")
    axs[1].set_ylabel("Time in Forward Pass (%)")
    axs[0].set_ylabel("Total Runtime/move (s)") 

    axs[0].grid(True)
    axs[1].grid(True)

    legend_elements = [
        Patch(facecolor="firebrick", label="mean"),
        Patch(facecolor="blue", label="max"),
    ]

    axs[0].legend(handles=legend_elements, loc="upper left")

    f.savefig("figures/core_dependency.png", bbox_inches="tight")


def graph_games(data: np.ndarray): 
    ### num games dependency

    f, axs = plt.subplots(2, 1, dpi=600, figsize=(6, 4.5))
    for run in data:
        if run.num_cores == 2:
            axs[0].scatter(run.num_games, np.mean(run.total_times), color="firebrick")
            axs[0].scatter(run.num_games, np.max(run.total_times), color="blue")
            axs[1].scatter(
                run.num_games,
                np.mean(100 * run.forward_times / run.total_times),
                color="forestgreen",
            )

        f.suptitle("Dependency on Number of Games Played")

    axs[1].set_xlabel("Number of Games played")
    axs[1].set_ylabel("Time in Forward Pass (%)")
    axs[0].set_ylabel("Total Runtime/move (s)") 

    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    legend_elements = [
        Patch(facecolor="firebrick", label="mean"),
        Patch(facecolor="blue", label="max"),
    ]

    axs[0].legend(handles=legend_elements, loc="upper left")

    f.savefig("figures/game_dependency.png", bbox_inches="tight")



if __name__ == "__main__":

    filename = "BENCHece.txt"

    runs_data = []
    current_block = []

    with open(filename, "r") as f:
        for line in f:
            if line[0] == "S":
                pass
            elif line[0] == "G" and len(current_block) > 0:
                runs_data.append(Run.from_block(current_block[:-1]))
                current_block = []
                current_block.append(line)
            elif line[0] == "T":
                current_block.append(line)
        runs_data.append(Run.from_block(current_block[:-1]))

    # graph_cores(np.array(runs_data))
    graph_games(runs_data)
