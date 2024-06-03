import matplotlib.pyplot as plt


class KeyBinds:
    key_to_str = lambda key: key.lower().replace("-", " + ")
    view_hotkeys = "Shift-k"
    switch_zoom_select = "Space"
    remove_background = "i"
    view_bleed_matrix = "b"
    view_background_norm = "n"
    view_bleed_matrix_calculation = "Shift-b"
    view_bled_codes = "g"
    view_all_gene_scores = "Shift-h"
    view_gene_efficiency = "e"
    view_gene_counts = "Shift-g"
    view_histogram_scores = "h"
    view_scaled_k_means = "k"
    view_colour_and_codes = "c"
    view_spot_intensities = "s"
    view_spot_colours_and_weights = "d"
    view_intensity_from_colour = "Shift-i"
    view_omp_coefficients = "o"


class ViewHotkeys:
    def __init__(self) -> None:
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        fig.suptitle("Hotkeys", size=20)
        ax.set_axis_off()
        text = f"""Toggle zoom/spot selection: {KeyBinds.key_to_str(KeyBinds.switch_zoom_select)}
                Remove background image: {KeyBinds.key_to_str(KeyBinds.remove_background)}
                View bleed matrix: {KeyBinds.key_to_str(KeyBinds.view_bleed_matrix)}
                View background, normalised: {KeyBinds.key_to_str(KeyBinds.view_background_norm)}
                View bleed matrix calculation: {KeyBinds.key_to_str(KeyBinds.view_bleed_matrix_calculation)}
                View bled codes: {KeyBinds.key_to_str(KeyBinds.view_bled_codes)}
                View all gene scores: {KeyBinds.key_to_str(KeyBinds.view_all_gene_scores)}
                View gene efficiencies: {KeyBinds.key_to_str(KeyBinds.view_gene_efficiency)}
                View gene counts: {KeyBinds.key_to_str(KeyBinds.view_gene_counts)}
                View score histogram: {KeyBinds.key_to_str(KeyBinds.view_histogram_scores)}
                View scaled k means: {KeyBinds.key_to_str(KeyBinds.view_scaled_k_means)}
                Compare spot colours and codes: {KeyBinds.key_to_str(KeyBinds.view_colour_and_codes)}
                View spot intensities: {KeyBinds.key_to_str(KeyBinds.view_spot_intensities)}
                View spot colours and weights: {KeyBinds.key_to_str(KeyBinds.view_spot_colours_and_weights)}
                View intensities calculation from colour: {KeyBinds.key_to_str(KeyBinds.view_intensity_from_colour)}
                View OMP coefficients: {KeyBinds.key_to_str(KeyBinds.view_omp_coefficients)}
                View OMP fit: {KeyBinds.key_to_str(KeyBinds.view_omp_fit)}
                View OMP score: {KeyBinds.key_to_str(KeyBinds.view_omp_score)}"""
        ax.text(
            0.1,
            0.5,
            text,
            fontdict={"size": 12, "verticalalignment": "center", "horizontalalignment": "center"},
            verticalalignment="center",
            horizontalalignment="left",
        )
        fig.show()
