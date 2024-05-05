import wandb
import matplotlib.pyplot as plt
api = wandb.Api()
run = api.run("/praveshkoirala/Rideshare Multi Episodic/runs/y5okgo99")
responsive = run.history()
run = api.run("/praveshkoirala/Rideshare Multi Episodic/runs/nr1g3894")
lagging = run.history()


def plot_df_data(df, output_png_filename, fields, colors, opacity, time_avg_steps,
                  title, legends, xmin, ymin, xmax, ymax, xlabel, ylabel):
    data = df
    plt.figure(figsize=(12, 6))

    for i, field in enumerate(fields):
        plt.plot(data[field], color=colors[i], alpha=opacity)

    for i, field in enumerate(fields):
        if time_avg_steps > 0:
            moving_avg = data[field].rolling(window=time_avg_steps).mean()
            exp_moving_avg = data[field].ewm(alpha=0.5, adjust=False).mean()

            plt.plot(exp_moving_avg, color=colors[i], alpha=1.0, label=legends[i])  # Full opacity for average

    # plt.title(title)
    plt.legend(fontsize=25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_png_filename, format='png', bbox_inches='tight')
    plt.close()

for run, df in [("Responsive", responsive), ("Lagging", lagging)]:

    # Plot Profits
    plot_df_data(
        df=df,
        output_png_filename=f'graphs/profit_{run}.png',
        fields=['profits_U', 'profits_L'],
        colors=['#DF672A', '#338DD8'],
        opacity=0.25,
        time_avg_steps=10,
        title=f'End of episode profits for U and L ({run} market)',
        legends=['U', 'L'],
        xmin=0,
        ymin=-100,
        xmax=500,
        ymax=200,
        xlabel="Epochs",
        ylabel="Profits"
    )
    for e in [('0', '1'), ('1','0')]:
        for p in ["U", "L"]:
            plot_df_data(
                df=df,
                output_png_filename=f'graphs/R{p}C{p}_{e[0]}_{e[1]}_{run}.png',
                fields=[f"R{p}_{e[0]}_{e[1]}", f"C{p}_{e[0]}_{e[1]}"],
                colors=['#DF672A', '#338DD8'],
                opacity=0.25,
                time_avg_steps=10,
                title=f'Rates/Commission for {p} on edge {e[0]}-{e[1]} ({run} market)',
                legends=['Rate', 'Commission'],
                xmin=0,
                ymin=0,
                xmax=500,
                ymax=20,
                xlabel="Epochs",
                ylabel="Amount"
            )
