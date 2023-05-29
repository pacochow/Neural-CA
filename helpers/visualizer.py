import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_animation(states, iterations, nSeconds, filename):

    fps = iterations/nSeconds

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )

    a = states[0]
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
    plt.axis('off')

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        im.set_array(states[i])
        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = iterations,
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Done!')
    