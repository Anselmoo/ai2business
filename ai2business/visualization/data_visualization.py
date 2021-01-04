"""Data Visualization Module: Visualization of data and its first principal properties."""
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from secrets import token_hex
from typing import Callable, Union

import matplotlib.pyplot as plt
import missingno as mss
import numpy as np
import pandas as pd
import seaborn as sns

from ai2business.visualization import style


class BuilderDataVisualization(ABC):
    """BuilderDataVisualization contains the abstract properties and methods.

    `BuilderDataVisualization` specifies the properties and methods for creating the
    different parts of the `DesignerDataVisualization` objects.

    Args:
        ABC (class): Helper class that provides a standard way to create an ABC using inheritance.
    """

    @abstractmethod
    def reset(self) -> None:
        """Abstract method of reset."""

    @abstractproperty
    def data_figure(self) -> None:
        """Abstract property of data_figure."""

    @abstractmethod
    def get_lineplot(self) -> None:
        """Abstract method of get_lineplot."""

    @abstractmethod
    def get_pointplot(self) -> None:
        """Abstract method of get_pointplot."""

    @abstractmethod
    def get_scatterplot(self) -> None:
        """Abstract method of get_scatterplot."""

    @abstractmethod
    def get_swarmplot(self) -> None:
        """Abstract method of get_swarmplot."""

    @abstractmethod
    def get_distributionplot(self) -> None:
        """Abstract method of get_distributionplot."""

    @abstractmethod
    def get_relationalplot(self) -> None:
        """Abstract method of get_relationalplot."""

    @abstractmethod
    def get_categoryplot(self) -> None:
        """Abstract method of get_categoryplot."""

    @abstractmethod
    def get_boxplot(self) -> None:
        """Abstract method of get_boxplot."""

    @abstractmethod
    def get_boxenplot(self) -> None:
        """Abstract method of get_boxenplot."""

    @abstractmethod
    def get_stripplot(self) -> None:
        """Abstract method of get_stripplot."""

    @abstractmethod
    def get_hexagonplot(self) -> None:
        """Abstract method of get_hexagonplot."""

    @abstractmethod
    def get_histogramplot(self) -> None:
        """Abstract method of get_histogramplot."""

    @abstractmethod
    def get_violinplot(self) -> None:
        """Abstract method of get_violinplot."""

    @abstractmethod
    def get_residualplot(self) -> None:
        """Abstract method of get_residualplot."""

    @abstractmethod
    def get_regressionplot(self) -> None:
        """Abstract method of get_regressionplot."""

    @abstractmethod
    def get_density_mapplot(self) -> None:
        """Abstract method of get_densitymapplot."""

    @abstractmethod
    def get_kerneldensity_mapplot(self) -> None:
        """Abstract method of get_kerneldensity_mapplot."""

    @abstractmethod
    def get_cluster_mapplot(self) -> None:
        """Abstract method of get_cluster_mapplot."""

    @abstractmethod
    def get_heatmapplot(self) -> None:
        """Abstract method of get_heatmapp."""

    @abstractmethod
    def get_correlationpplot(self) -> None:
        """Abstract method of get_correlationpplot."""

    @abstractmethod
    def get_diagonal_correlationpplot(self) -> None:
        """Abstract method of get_diagonal_correlationpplot."""

    def get_regression_mapplot(self) -> None:
        """Abstract method of get_marginalplot."""

    @abstractmethod
    def get_pairmapplot(self) -> None:
        """Abstract method of get_pairmapplot."""

    @abstractmethod
    def get_complex_pairmapplot(self) -> None:
        """Abstract method of get_complex_pairmapplot."""

    @abstractmethod
    def get_nullity_matrix(self) -> None:
        """Abstract method of get_nullity_matrix."""

    @abstractmethod
    def get_nullity_bar(self) -> None:
        """Abstract method of get_nullity_bar."""

    @abstractmethod
    def get_nullity_heatmap(self) -> None:
        """Abstract method of get_nullity_heatmap."""

    @abstractmethod
    def get_nullity_dendrogram(self) -> None:
        """Abstract method of get_nullity_dendrogram."""


class DataVisualizationProduct:

    """DataVisualizationProduct contains the dictionary and the return value of it."""

    def __init__(self) -> None:
        """Initialization of DataVisualizationProduct."""
        self.product_parts = {}

    def add_product(self, key: Callable, value: plt.subplot) -> None:
        """Add the components of the data visualization to the dictionary.

        Args:
            key (Callable): Used data visualization search function
            value (plt.subplot): Return value as `matplotlib`-class of subplots.
        """
        self.product_parts[key.__name__] = value

    @property
    def list_product_parts(self) -> str:
        """List of the product parts in the dictionary."""
        return f"Product parts: {', '.join(self.product_parts)}"

    @property
    def return_product(self) -> dict:
        """Returns the product as a dictionary

        Returns:
            dict: The product dictionary contains the product and ist function name as `key`.
        """
        return self.product_parts

    def save_all_figures(self, folder: str = None):
        """Saving all figures of the product dictionary.

        All figures will be saved at `.` or in subfoalder as a combination of the current
        `key` and a 16 Bytes hex-token.

        Args:
            folder (str, optional): Name of the subfolder, which will be created if not exists. Defaults to None.
        """
        path = Path(".")
        for key, value in self.product_parts.items():
            if folder:
                path = Path(folder)
                path.mkdir(exist_ok=True)
            value.savefig(path.joinpath(f"{key}_{token_hex(16)}.png"))


class DesignerDataVisualization(BuilderDataVisualization):
    """DesignerDataVisualization contains the specific implementation of
    `BuilderDataVisualization`.

    `DesignerDataVisualization` contains the specific implementation of
    `BuilderDataVisualization` based on the external libraries:

    1. [https://github.com/ResidentMario/missingno](https://github.com/ResidentMario/missingno)
    2. [ttps://github.com/mwaskom/seaborn](https://github.com/mwaskom/seaborn)

    Args:
        BuilderDataVisualization (class): Abstract class that provides the implementations of the properties and methods.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dark_mode: bool = False,
        grid: bool = False,
        figsize: tuple = (12, 8),
        dpi: int = 300,
        x_label: str = None,
        y_label: str = None,
        hue: str = None,
        palette: Union[str, list, dict] = None,
    ) -> None:
        """Initialization of DesignerDataVisualization.

        Args:
            df (pd.DataFrame): pandas DataFrame.
            dark_mode (bool, optional): Switch to dark mode. Defaults to False.
            grid (bool, optional): Activate grids in plots. Defaults to False.
            figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
            dpi (int, optional): Resolution of the figure. Defaults to 300.
            x_label (str, optional): Name of the column name for the `x-axis`. Defaults to None.
            y_label (str, optional): Name of the column name for the `y-axis`. Defaults to None.
            hue (str, optional): Name of the column name for the seperating the results to the uniques once. Defaults to None.
            palette (Union[str, list,  dict], optional): The `str`, `list`, or `dict` of colors or continuous colormap, which defines the color palette. Defaults to None.


        !!! note "Appearance-Modes"

            1. Light-Mode without Grid

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/appearance/get_lineplot_805fabb7bbea3a9e807d3c2444bbaa4e.png?raw=true){: loading=lazy }

            2. Light-Mode with Grid

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/appearance/get_lineplot_520cb6360c9fb9c61977a303213f1340.png?raw=true){: loading=lazy }

            3. Dark-Mode without Grid

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/appearance/get_lineplot_6ef44df80722a1c466d0a4c47b8f2433.png?raw=true){: loading=lazy }

            4. Dark-Mode with Grid

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/appearance/get_lineplot_65d5fcb713a18d25b1c1f6b504836776.png?raw=true){: loading=lazy }


        !!! note "Figuere Size"

            Due to the default settings of `missingno`, the figure size has to be defined in advance.
        """
        self.df = df
        self.x_label = x_label
        self.y_label = y_label
        self.hue = hue
        self.palette = palette
        self.figsize = figsize
        _ = plt.figure(figsize=self.figsize, dpi=dpi)
        if dark_mode:
            self.style = style.dark_mode(grid=grid)
        else:
            self.style = style.light_mode(grid=grid)
        self.reset()

    def reset(self) -> None:
        """Reset the product to empty."""
        self._product = DataVisualizationProduct()

    @property
    def data_figure(self) -> DataVisualizationProduct:
        """Return the results of the figure class of the data visualization.

        Returns:
            DataVisualizationProduct (class): DataVisualizationProduct contains the dictionary and the return value of it.
        """
        product = self._product
        self.reset()
        return product

    def get_lineplot(self, **kwargs) -> None:
        """Get a line plot."""
        self._product.add_product(
            key=self.get_lineplot,
            value=sns.lineplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_pointplot(self, **kwargs) -> None:
        """Get a point plot."""
        self._product.add_product(
            key=self.get_pointplot,
            value=sns.pointplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_scatterplot(self, **kwargs) -> None:
        """Get a scatter plot."""
        self._product.add_product(
            key=self.get_scatterplot,
            value=sns.scatterplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_swarmplot(self, **kwargs) -> None:
        """Get a swarm plot."""
        self._product.add_product(
            key=self.get_swarmplot,
            value=sns.swarmplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_distributionplot(self, **kwargs) -> None:
        """Get a distribution plot."""
        self._product.add_product(
            key=self.get_distributionplot,
            value=sns.displot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ),
        )

    def get_relationalplot(self, **kwargs) -> None:
        """Get a relational plot."""
        self._product.add_product(
            key=self.get_relationalplot,
            value=sns.relplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ),
        )

    def get_categoryplot(self, **kwargs) -> None:
        """Get a category plot."""
        self._product.add_product(
            key=self.get_categoryplot,
            value=sns.catplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ),
        )

    def get_boxplot(self, **kwargs) -> None:
        """Get a box plot."""
        self._product.add_product(
            key=self.get_boxplot,
            value=sns.boxplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_boxenplot(self, **kwargs) -> None:
        """Get a multi box plot."""
        self._product.add_product(
            key=self.get_boxenplot,
            value=sns.boxenplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_stripplot(self, **kwargs) -> None:
        """Get a strip plot."""
        self._product.add_product(
            key=self.get_stripplot,
            value=sns.stripplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_hexagonplot(self, **kwargs) -> None:
        """Get a hexagon plot."""
        self._product.add_product(
            key=self.get_hexagonplot,
            value=sns.jointplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                kind="hex",
                **kwargs,
            ),
        )

    def get_histogramplot(self, **kwargs) -> None:
        """Get a histogram plot."""
        self._product.add_product(
            key=self.get_histogramplot,
            value=sns.histplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_violinplot(self, **kwargs) -> None:
        """Get a violinplot plot."""
        self._product.add_product(
            key=self.get_violinplot,
            value=sns.violinplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ).get_figure(),
        )

    def get_residualplot(self, **kwargs) -> None:
        """Get a residual plot."""
        self._product.add_product(
            key=self.get_residualplot,
            value=sns.residplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                **kwargs,
            ).get_figure(),
        )

    def get_regressionplot(self, **kwargs) -> None:
        """Get a regression plot."""
        self._product.add_product(
            key=self.get_regressionplot,
            value=sns.lmplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ),
        )

    def get_density_mapplot(self, **kwargs) -> None:
        """Get a density map plot."""
        self._product.add_product(
            key=self.get_density_mapplot,
            value=sns.jointplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                kind="kde",
            ),
        )

    def get_kerneldensity_mapplot(self, **kwargs) -> None:
        """Get a kernel density map plot."""
        self._product.add_product(
            key=self.get_kerneldensity_mapplot,
            value=sns.jointplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                kind="kde",
                **kwargs,
            ),
        )

    def get_cluster_mapplot(
        self, method: str = "pearson", min_periods: int = 1, **kwargs
    ) -> None:
        """Get a cluster map plot.

        Args:
            method (str, optional): Method of the correlation type ('pearson', 'kendall', 'spearman' or callable method of correlation). Defaults to "pearson".
            min_periods (int, optional): Minimum number of observations required per pair of columns to have a valid result. Defaults to 1.
        """
        self._product.add_product(
            key=self.get_cluster_mapplot,
            value=sns.clustermap(
                data=self.df.corr(method=method, min_periods=min_periods), **kwargs
            ),
        )

    def get_heatmapplot(self, **kwargs) -> None:
        """Get a heatmap plot."""
        self._product.add_product(
            key=self.get_cluster_mapplot,
            value=sns.heatmap(data=self.df, **kwargs).get_figure(),
        )

    def get_correlationpplot(
        self, method: str = "pearson", min_periods: int = 1, **kwargs
    ) -> None:
        """Get a correlation map plot.

        Args:
            method (str, optional): Method of the correlation type ('pearson', 'kendall', 'spearman' or callable method of correlation). Defaults to "pearson".
            min_periods (int, optional): Minimum number of observations required per pair of columns to have a valid result. Defaults to 1.
        """
        self._product.add_product(
            key=self.get_correlationpplot,
            value=sns.relplot(
                data=self.df.corr(method=method, min_periods=min_periods),
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ),
        )

    def get_diagonal_correlationpplot(
        self, method: str = "pearson", min_periods: int = 1, **kwargs
    ) -> None:
        """Get a correlation map plot with lower non-diagonal elements.

        Args:
            method (str, optional): Method of the correlation type ('pearson', 'kendall', 'spearman' or callable method of correlation). Defaults to "pearson".
            min_periods (int, optional): Minimum number of observations required per pair of columns to have a valid result. Defaults to 1.
        """
        _corr = self.df.corr(method=method, min_periods=min_periods)
        _mask = np.triu(np.ones_like(_corr, dtype=bool))
        self._product.add_product(
            key=self.get_cluster_mapplot,
            value=sns.heatmap(data=_corr, mask=_mask, **kwargs).get_figure(),
        )

    def get_pairmapplot(self, **kwargs) -> None:
        """Get a pair plot."""
        self._product.add_product(
            key=self.get_pairmapplot,
            value=sns.lmplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                **kwargs,
            ),
        )

    def get_complex_pairmapplot(self, **kwargs) -> None:
        """Get a complex pair plot.

        !!! note "About Complex Pair Plots"
            The complex pair consits of three different types of subplots:

            1. A distribution plot on the diagonal.
            2. A kernel distribution map on the lower non-diagonal.
            3. A scatter plot in combination with a linear regression on the upper non-diagonal.
        """
        grid = sns.PairGrid(self.df, hue=self.hue, palette=self.palette, **kwargs)
        grid.map_upper(sns.regplot)
        grid.map_lower(sns.kdeplot)
        grid.map_diag(sns.kdeplot)
        self._product.add_product(
            key=self.get_pairmapplot,
            value=grid,
        )

    def get_regression_mapplot(self, **kwargs) -> None:
        """Get a regression map plot."""
        self._product.add_product(
            key=self.get_regression_mapplot,
            value=sns.jointplot(
                data=self.df,
                x=self.x_label,
                y=self.y_label,
                hue=self.hue,
                palette=self.palette,
                kind="reg",
            ),
        )

    def get_nullity_matrix(
        self, n_columns: int = 0, per_columns: float = 0.0, **kwargs
    ) -> None:
        """A bar matrix visualization of the nullity of the given DataFrame.

        Args:
            n_columns (int, optional): The cap on the number of columns to include in the filtered DataFrame. Defaults to 0.
            per_columns (float, optional): The cap on the percentage fill of the columns in the filtered DataFrame. Defaults to 0.0.
        """
        self._product.add_product(
            key=self.get_nullity_matrix,
            value=mss.matrix(
                df=self.df, n=n_columns, p=per_columns, figsize=self.figsize, **kwargs
            ).get_figure(),
        )

    def get_nullity_bar(
        self, n_columns: int = 0, per_columns: float = 0.0, **kwargs
    ) -> None:
        """A bar chart visualization of the nullity of the given DataFrame.

        Args:
            n_columns (int, optional): The cap on the number of columns to include in the filtered DataFrame. Defaults to 0.
            per_columns (float, optional): The cap on the percentage fill of the columns in the filtered DataFrame. Defaults to 0.0.
        """
        self._product.add_product(
            key=self.get_nullity_bar,
            value=mss.bar(
                df=self.df, n=n_columns, p=per_columns, figsize=self.figsize, **kwargs
            ).get_figure(),
        )

    def get_nullity_heatmap(
        self,
        n_columns: int = 0,
        per_columns: float = 0.0,
        cmap: str = "seismic",
        **kwargs,
    ) -> None:
        """A heatmap chart visualization of the nullity of the given DataFrame.

        Args:
            n_columns (int, optional): The cap on the number of columns to include in the filtered DataFrame. Defaults to 0.
            per_columns (float, optional): The cap on the percentage fill of the columns in the filtered DataFrame. Defaults to 0.0.
            cmap (str, optional): The color of the heatmap. Defaults to "seismic".
        """
        self._product.add_product(
            key=self.get_nullity_heatmap,
            value=mss.heatmap(
                df=self.df,
                n=n_columns,
                p=per_columns,
                figsize=self.figsize,
                cmap=cmap,
                **kwargs,
            ).get_figure(),
        )

    def get_nullity_dendrogram(
        self, n_columns: int = 0, per_columns: float = 0.0, **kwargs
    ) -> None:
        """Generates the nullity dendrogram."""
        self._product.add_product(
            key=self.get_nullity_dendrogram,
            value=mss.dendrogram(
                df=self.df, n=n_columns, p=per_columns, figsize=self.figsize, **kwargs
            ).get_figure(),
        )


class DataVisualization:
    """DataVisualization is in charge of executing the functions.

    During the execution, `DataVisualization` can construct several product
    variations using the same building steps.

    !!! example "General introduction into using plot-functions!"
        ```python
        >>> from ai2business.macros import oneliner as one
        >>> from ai2business.visualization import data_visualization as dav

        >>> df_dict_years = one.TrendSearch.four_step_search(keyword_list=[ "2017", "2018", "2019", "2020", "2021", ])

        >>> data = dav.DataVisualization()
        >>> builder = dav.DesignerDataVisualization(df_dict_years["get_interest_over_time"])
        >>> data.builder = builder

        # Here any kind of plot function can be called
        >>> data.lineplot()
        >>> builder.data_figure.save_all_figures(folder=folder)
        ```
    !!! tip "Activating hidden functions of the `seaborn`-module!"
        Due to the seaborn module's complexity, only the significant __four__
        variables (x_label, y_label, hue, palette) are defined in
        `DesignerDataVisualization`. However, all other seaborn-module options
        can be individual activated by`**kwargs` in each function separately.
    """

    def __init__(self) -> None:
        """Initialize a fresh and empty builder."""
        self._builder = None

    @property
    def builder(self) -> BuilderDataVisualization:
        """Builder as a property with value None.

        Returns:
            BuilderDataVisualization: A builder class, that contains the abstract properties and methods.
        """
        return self._builder

    @builder.setter
    def builder(self, builder: BuilderDataVisualization) -> property:
        """Sets the builder according to BuilderDataVisualization.

        Args:
            builder (BuilderDataVisualization): A builder class, that contains the abstract properties and methods.

        Returns:
            property: A method or property of `BuilderDataVisualization`.
        """
        self._builder = builder

    def visual_missing_data(
        self,
        n_columns: int = 0,
        per_columns: float = 0.0,
        cmap: str = "seismic",
        **kwargs,
    ) -> None:
        """Visualizing possible missing data.

        Args:
            n_columns (int, optional): The cap on the number of columns to include in the filtered DataFrame. Defaults to 0.
            per_columns (float, optional): The cap on the percentage fill of the columns in the filtered DataFrame. Defaults to 0.0.
            cmap (str, optional): The color of the heatmap. Defaults to "seismic".

        !!! note "Visualization of nullity results, respectively, missing values!"

            Possible missing data will be visualized by:

            1. nullity matrix highlights out patterns and structures in data completion.

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/nullity/get_nullity_matrix_bc6141f519e4eea9e6a0d5fe84115d65.png?raw=true){: loading=lazy }

            2. nullity bar shows the available data as a series of single bars.

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/nullity/get_nullity_bar_9918e972a769781d322ba1e18fe8f86c.png?raw=true){: loading=lazy }

            3. nullity heatmap point out the correlation between the presence and absence data.

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/nullity/get_nullity_heatmap_d46e69f559bd7b713b7a6c3ceb0f9968.png?raw=true){: loading=lazy }

            4. nullity dendrogram visualize the correlate variable completion.

                ![Placeholder](https://github.com/AI2Business/ai2business/blob/main/docs/images/nullity/get_nullity_dendrogram_15a148832cd43d6d249d56671a0fab6f.png?raw=true){: loading=lazy }

            For more information see: [https://github.com/ResidentMario/missingno](https://github.com/ResidentMario/missingno)
        """
        self.builder.get_nullity_matrix(
            n_columns=n_columns, per_columns=per_columns, **kwargs
        )
        self.builder.get_nullity_bar(
            n_columns=n_columns, per_columns=per_columns, **kwargs
        )
        self.builder.get_nullity_heatmap(
            n_columns=n_columns, per_columns=per_columns, cmap=cmap, **kwargs
        )
        self.builder.get_nullity_dendrogram(
            n_columns=n_columns, per_columns=per_columns, **kwargs
        )

    @property
    def initialization_figure(self) -> None:
        """Initialization of the figure.

        !!! danger "Reminder"
            It is very important that if figure has to be first deleted,
            otherwise the figures can be overlapped.
        """
        plt.cla()
        plt.clf()

    def lineplot(self, **kwargs) -> None:
        """Create a given line plot based on seaborn."""
        self.initialization_figure
        self.builder.get_lineplot(**kwargs)

    def pointplot(self, **kwargs) -> None:
        """Create a given point plot based on seaborn."""
        self.initialization_figure
        self.builder.get_pointplot(**kwargs)

    def scatterplot(self, **kwargs) -> None:
        """Create a given scatter plot based on seaborn."""
        self.initialization_figure
        self.builder.get_scatterplot(**kwargs)

    def swarmplot(self, **kwargs) -> None:
        """Create a given swarm plot based on seaborn."""
        self.initialization_figure
        self.builder.get_swarmplot(**kwargs)

    def distributionplot(self, **kwargs) -> None:
        """Create a given distribution plot based on seaborn."""
        self.builder.get_distributionplot(**kwargs)

    def relationalplot(self, **kwargs) -> None:
        """Create a given relational plot based on seaborn."""
        self.initialization_figure
        self.builder.get_relationalplot(**kwargs)

    def categoryplot(self, **kwargs) -> None:
        """Create a given category plot based on seaborn."""
        self.initialization_figure
        self.builder.get_categoryplot(**kwargs)

    def boxplot(self, multiboxen: bool = False, **kwargs) -> None:
        """Create a given box plot based on seaborn.

        Args:
            multiboxen (bool, optional): Allows to draw multi boxen per object. Defaults to False.
        """
        self.initialization_figure
        if multiboxen:
            self.builder.get_boxenplot(**kwargs)
        else:
            self.builder.get_boxplot(**kwargs)

    def stripplot(self, **kwargs) -> None:
        """Create a given strip plot based on seaborn."""
        self.initialization_figure
        self.builder.get_stripplot(**kwargs)

    def hexagonplot(self, **kwargs) -> None:
        """Create a given hexagon plot based on seaborn."""
        self.initialization_figure
        self.builder.get_hexagonplot(**kwargs)

    def histogramplot(self, **kwargs) -> None:
        """Create a given histogram plot based on seaborn."""
        self.initialization_figure
        self.builder.get_histogramplot(**kwargs)

    def violinplot(self, **kwargs) -> None:
        """Create a given violin plot based on seaborn."""
        self.initialization_figure
        self.builder.get_violinplot(**kwargs)

    def residualplot(self, **kwargs) -> None:
        """Create a given residual plot based on seaborn."""
        self.initialization_figure
        self.builder.get_residualplot(**kwargs)

    def regressionplot(self, map: bool = False, **kwargs) -> None:
        """Create a given regression plot based on seaborn.

        Args:
            map (bool, optional): Creates the regression plot as map. Defaults to False.
        """
        self.initialization_figure
        if map:
            self.builder.get_regression_mapplot(**kwargs)
        else:
            self.builder.get_regressionplot(**kwargs)

    def densitymap(self, kde: bool = False, **kwargs) -> None:
        """Create a given density map based on seaborn.
        Args:
            kde (bool, optional): Plots the density as kernel density. Defaults to False.
        """
        self.initialization_figure

        if kde:
            self.builder.get_kerneldensity_mapplot(**kwargs)
        else:
            self.builder.get_density_mapplot(**kwargs)

    def clustermap(
        self, method: str = "pearson", min_periods: int = 1, **kwargs
    ) -> None:
        """Create a given cluster map based on seaborn.

        Args:
            method (str, optional): Method of the correlation type ('pearson', 'kendall', 'spearman' or callable method of correlation). Defaults to "pearson".
            min_periods (int, optional): Minimum number of observations required per pair of columns to have a valid result. Defaults to 1.
        """
        self.initialization_figure
        self.builder.get_cluster_mapplot(
            method=method, min_periods=min_periods, **kwargs
        )

    def heatmap(self, **kwargs) -> None:
        """Create a given heat map based on seaborn."""
        self.initialization_figure
        self.builder.get_heatmapplot(**kwargs)

    def correlationmap(
        self,
        diagonal: bool = False,
        method: str = "pearson",
        min_periods: int = 1,
        **kwargs,
    ) -> None:
        """Create a given correlation map based on seaborn.

        Args:
            diagonal (bool, optional): Only the lower diagonal elements will be plotted. Defaults to False.
            method (str, optional): Method of the correlation type ('pearson', 'kendall', 'spearman' or callable method of correlation). Defaults to "pearson".
            min_periods (int, optional): Minimum number of observations required per pair of columns to have a valid result. Defaults to 1.
        """
        self.initialization_figure
        if diagonal:
            self.builder.get_diagonal_correlationpplot(
                method=method, min_periods=min_periods, **kwargs
            )
        else:
            self.builder.get_correlationpplot(
                method=method, min_periods=min_periods, **kwargs
            )

    def pairmap(self, complex: bool = False, **kwargs):
        """Create a pair map based on seaborn.

        Args:
            complex (bool, optional): Turn on the `get_complex_pairmapplot`. Defaults to False.
        """
        self.initialization_figure
        if complex:
            self.builder.get_complex_pairmapplot(**kwargs)
        else:
            self.builder.get_pairmapplot(**kwargs)
