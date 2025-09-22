import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from math import floor

class ColourTempCurve:
    """
    Class that maps colour temperature to red and blue values using a piecewise linear function
    in 2 dimensions.
    """
    def __init__(self, points : List[float], sensitivity_r : float, sensitivity_b : float) -> None:
        """
        Initialise a 2D piecewise linear function from a list of points. The points come in groups of three,
        representing a colour temperature and a pair of red and blue values.

        Args:
            points (List[float]): List of points in the format [colour_temp1, red1, blue1, colour_temp2, red2, blue2, ...]
        """
        self.colour_temp = np.array(points[0::3])
        self.red = np.array(points[1::3])
        self.blue = np.array(points[2::3])
        self.red *= sensitivity_r
        self.blue *= sensitivity_b

    def eval(self, colour_temp : float) -> np.ndarray:
        """
        Evaluate the function at a given colour temperature to get red and blue values.

        Args:
            colour_temp (float): Colour temperature in Kelvin

        Returns:
            np.ndarray: Array of red and blue values
        """
        red = np.interp(colour_temp, self.colour_temp, self.red)
        blue = np.interp(colour_temp, self.colour_temp, self.blue)
        return np.array([red, blue])

    def invert(self, red_blue : np.ndarray) -> float:
        """
        Invert the function to find the colour temperature where the given point lies
        on a line perpendicular to the curve at that temperature.

        Args:
            red_blue (np.ndarray): Array of red and blue values

        Returns:
            float: Colour temperature in Kelvin
        """
        # Initial guess
        red, blue = red_blue
        colour_temp_red = np.interp(red, self.red[::-1], self.colour_temp[::-1])
        colour_temp_blue = np.interp(blue, self.blue, self.colour_temp)
        initial_temp = (colour_temp_red + colour_temp_blue) / 2

        # Use optimization to find temperature where point lies on perpendicular line
        def objective(temp):
            # Get curve point and transverse direction at this temperature
            curve_point = self.eval(temp)
            transverse = self.transverse(temp)
            if transverse is None:
                return 1000000

            # Vector from curve point to target point
            to_target = red_blue - curve_point

            # Project onto transverse direction and find perpendicular component
            # We want the component perpendicular to the transverse to be zero
            parallel_component = np.dot(to_target, transverse)
            perpendicular_component = to_target - parallel_component * transverse

            # Return squared magnitude of perpendicular component
            return np.dot(perpendicular_component, perpendicular_component)

        # Simple ternary search for the minimum
        left = self.colour_temp.min()
        right = self.colour_temp.max()
        tolerance = 1.0  # 1K tolerance

        # Ternary search
        while right - left > tolerance:
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3

            if objective(m1) > objective(m2):
                left = m1
            else:
                right = m2

        result_temp = (left + right) / 2

        # Validate result - if objective is too large, fall back to original method
        if objective(result_temp) > 0.01:  # threshold for "close enough"
            return initial_temp

        return result_temp

    def invert_fast(self, red_blue : np.ndarray) -> float:
        """
        Invert the function to find the colour temperature for a given pair of r and b values.
        """
        red, blue = red_blue
        colour_temp_red = np.interp(red, self.red[::-1], self.colour_temp[::-1])
        colour_temp_blue = np.interp(blue, self.blue, self.colour_temp)
        return (colour_temp_red + colour_temp_blue) / 2

    def tangent(self, temp: float) -> np.ndarray | None:
        """
        Find a tangent line to the ct curve at a given colour temperature.
        Returns a vector equivalent to an increase of 1K.
        It is not a unit vector.
        """
        diff = 10
        red, blue = self.eval(temp - diff)
        red2, blue2 = self.eval(temp + diff)
        red_diff = red2 - red
        blue_diff = blue2 - blue
        tangent_vector = np.array([red_diff, blue_diff])
        norm = np.linalg.norm(tangent_vector)
        if norm == 0:
            return None
        else:
            return tangent_vector / (diff * 2)

    def transverse(self, temp: float) -> np.ndarray | None:
        """
        Find a perpendicular line to the ct curve at a given colour temperature.
        """
        diff = 10
        red, blue = self.eval(temp - diff)
        red2, blue2 = self.eval(temp + diff)
        red_diff = red2 - red
        blue_diff = blue2 - blue
        transverse_vector = np.array([blue_diff, -red_diff])
        norm = np.linalg.norm(transverse_vector)
        if norm == 0:
            return None
        else:
            return transverse_vector / norm

    def invert_with_transverse_multiple(self, red_blue: np.ndarray, fast: bool = False) -> Tuple[float, float]:
        """
        Convert a pair of red and blue gains to:
        - the colour temperature (K)
        - the scalar multiple along the unit transverse at that temperature

        Such that:
            red_blue ≈ eval(temp) + multiple * transverse(temp)

        Args:
            red_blue (np.ndarray): Array-like [red, blue] gains
            fast (bool): If True, use a faster but approximate temperature inversion

        Returns:
            Tuple[float, float]: (colour_temperature_K, transverse_multiple)
        """
        red_blue_np = np.array(red_blue, dtype=float)

        # Temperature at which the point lies on a line perpendicular to the curve
        if fast:
            temp = self.invert_fast(red_blue_np)
        else:
            temp = self.invert(red_blue_np)

        curve_point = self.eval(temp)
        transverse_unit = self.transverse(temp)

        if transverse_unit is None:
            return float(temp), 0.0

        # Projection of delta onto unit transverse gives the required multiple
        delta = red_blue_np - curve_point
        multiple = float(np.dot(delta, transverse_unit))

        return float(temp), multiple

class Tuning:
    """
    A class to hold a Raspberry Pi cameratuning file. Raspberry Pi 5 style tuning files should
    be used.
    """

    def __init__(self, json_config : str) -> None:
        """
        Initialise a tuning from a JSON configuration.

        Args:
            json_config (str): Path to the JSON configuration file
        """
        self.json_config = json_config

        self.colour_temp_curve = ColourTempCurve(self.get_algorithm("awb")["ct_curve"], self.get_algorithm("awb")["sensitivity_r"], self.get_algorithm("awb")["sensitivity_b"])

    @staticmethod
    def load(json_file : Union[str, Path]) -> 'Tuning':
        """
        Load the tuning from a JSON file.

        Args:
            json_file (Union[str, Path]): Path to the JSON configuration file

        Returns:
            Tuning: The tuning object
        """
        with open(json_file, 'r') as f:
            config = json.load(f)
        return Tuning(config)

    SEARCH_PATH = [Path("."), Path(__file__).resolve().parent / "tunings"]

    @staticmethod
    def find(sensor : str) -> Union[str, Path]:
        """
        Find the tuning for a given sensor, checking the folders listed in SEARCH_PATH.

        Args:
            sensor (str): Sensor model name

        Returns:
            Union[str, Path]: Path to the tuning file
        """
        for path in Tuning.SEARCH_PATH:
            tuning_file = path / f"{sensor}.json"
            if tuning_file.exists():
                return tuning_file

        raise FileNotFoundError(f"Tuning file for {sensor} not found")

    def get_algorithm(self, name: str) -> Dict[str, Any]:
        """
        Get algorithm configuration by name.
        Searches the 'algorithms' list for a dictionary with a key that ends with the given name.

        Args:
            name (str): Name of the algorithm to find

        Returns:
            Dict[str, Any]: The algorithm configuration dictionary

        Raises:
            KeyError: If the algorithm is not found
        """
        if "algorithms" not in self.json_config:
            raise KeyError("No 'algorithms' section found in configuration")

        algorithms = self.json_config["algorithms"]
        for algorithm in algorithms:
            for key, value in algorithm.items():
                if key.endswith(f".{name}"):
                    return value

        raise KeyError(f"Algorithm '{name}' not found in configuration")

    def get_colour_values(self, colour_temp: float) -> np.ndarray:
        """
        Get the colour values for a given colour temperature.

        Args:
            colour_temp (float): Colour temperature in Kelvin

        Returns:
            np.ndarray: Array of red and blue values
        """
        return self.colour_temp_curve.eval(colour_temp)

    def get_colour_temp(self, red_blue : np.ndarray) -> float:
        """
        Get the colour temperature for a given pair of red and blue values.

        Args:
            red_blue (np.ndarray): Array of red and blue values

        Returns:
            float: Colour temperature in Kelvin
        """
        return self.colour_temp_curve.invert(red_blue)

    def get_black_level(self, bits : int = 16) -> int:
        """
        Get the black level for the camera.

        Args:
            bits (int): Number of bits in the final black level output value (default 16)

        Returns:
            int: The black level
        """
        # The value in the file is always in 16 bits.
        return self.get_algorithm("black_level")["black_level"] >> (16 - bits)

    @staticmethod
    def _interpolate_table(colour_temp: float, tables : List[Dict[str, Any]]) -> np.ndarray:
        """
        Interpolate a table for a given colour temperature.
        """
        if colour_temp <= tables[0]["ct"]:
            return np.array(tables[0]["table"]).reshape(32, 32)
        elif colour_temp >= tables[-1]["ct"]:
            return np.array(tables[-1]["table"]).reshape(32, 32)

        # Find the two tables that bracket the given colour temperature
        for table, next_table in zip(tables[:-1], tables[1:]):
            if table["ct"] <= colour_temp and next_table["ct"] >= colour_temp:
                alpha = (colour_temp - table["ct"]) / (next_table["ct"] - table["ct"])
                return (alpha * np.array(next_table["table"]) + (1 - alpha) * np.array(table["table"])).reshape(32, 32)

        raise RuntimeError("Internal error: failed to interpolate LSC tables - should not happen")

    def get_lsc_tables(self, colour_temp: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get LSC tables for a given colour temperature.

        Args:
            colour_temp (float): Colour temperature in Kelvin

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The LSC tables for the R, G, and B channels
        """
        lsc_config = self.get_algorithm("alsc")
        cr_table = Tuning._interpolate_table(colour_temp, lsc_config["calibrations_Cr"])
        cb_table = Tuning._interpolate_table(colour_temp, lsc_config["calibrations_Cb"])
        luminance_table = np.array(lsc_config["luminance_lut"]).reshape(32, 32)
        luminance_strength = lsc_config.get("luminance_strength", 1.0)
        luminance_table = (luminance_table - 1.0) * luminance_strength + 1.0

        r_table = cr_table * luminance_table
        g_table = luminance_table
        b_table = cb_table * luminance_table

        return r_table / r_table.min(), g_table / g_table.min(), b_table / b_table.min()

    def get_gamma_curve(self) -> Tuple[List[int], List[int]]:
        """
        Get the gamma curve for the camera. Return two lists, one for the x values and one for the y values.
        """
        contrast_config = self.get_algorithm("contrast")
        gamma_curve = contrast_config["gamma_curve"]
        return (gamma_curve[0::2], gamma_curve[1::2])

    @staticmethod
    def _interpolate_table_ccm(colour_temp: float, tables : List[Dict[str, Any]]) -> np.ndarray:
        """
        Interpolate a table for the CCM for a given colour temperature.
        """
        if colour_temp <= tables[0]["ct"]:
            return np.array(tables[0]["ccm"]).reshape(3, 3)
        elif colour_temp >= tables[-1]["ct"]:
            return np.array(tables[-1]["ccm"]).reshape(3, 3)

        # Find the two tables that bracket the given colour temperature
        for table, next_table in zip(tables[:-1], tables[1:]):
            if table["ct"] <= colour_temp and next_table["ct"] >= colour_temp:
                alpha = (colour_temp - table["ct"]) / (next_table["ct"] - table["ct"])
                return (alpha * np.array(next_table["ccm"]) + (1 - alpha) * np.array(table["ccm"])).reshape(3, 3)

        raise RuntimeError("Internal error: failed to interpolate CCM tables - should not happen")

    def get_ccm(self, colour_temp: float) -> np.ndarray:
        """
        Get the CCM for a given colour temperature.
        """
        ccm_config = self.get_algorithm("ccm")
        return Tuning._interpolate_table_ccm(colour_temp, ccm_config["ccms"])

    def get_reference_lux(self) -> float:
        """
        Get the lux.
        """
        return self.get_algorithm("lux")["reference_lux"]

    def get_reference_gain(self) -> float:
        """
        Get the reference gain.
        """
        return self.get_algorithm("lux")["reference_gain"]

    def get_reference_shutter_speed(self) -> float:
        """
        Get the reference shutter speed.
        """
        return self.get_algorithm("lux")["reference_shutter_speed"]

    def get_reference_Y(self) -> float:
        """
        Get the reference Y.
        """
        return self.get_algorithm("lux")["reference_Y"]

    def get_reference_aperture(self) -> float:
        """
        Get the reference aperture.
        """
        return self.get_algorithm("lux")["reference_aperture"]

    def calculate_lux(self, Y: float, gain: float, aperture: float, shutter_speed: float) -> float:
        """
        Calculate the lux.
        """
        if aperture is None:
            aperture = self.get_reference_aperture()
        if shutter_speed is None:
            shutter_speed = self.get_reference_shutter_speed()
        if gain is None:
            gain = self.get_reference_gain()

        #print("Calculating lux: ")
        #print(f"Y: {Y}, reference Y: {self.get_reference_Y()}, ratio: {Y / self.get_reference_Y()}")
        #print(f"Aperture: {aperture}, reference aperture: {self.get_reference_aperture()}, ratio: {self.get_reference_aperture() / aperture}")
        #print(f"Shutter speed: {shutter_speed}, reference shutter speed: {self.get_reference_shutter_speed()}, ratio: {self.get_reference_shutter_speed() / shutter_speed}")
        #print(f"Gain: {gain}, reference gain: {self.get_reference_gain()}, ratio: {self.get_reference_gain() / gain}")

        lux = (self.get_reference_lux()
               * (Y / self.get_reference_Y())
               * (self.get_reference_aperture() / aperture)
               * (self.get_reference_shutter_speed() / shutter_speed)
               * (self.get_reference_gain() / gain))

        #print(f"Lux: {lux}, reference lux: {self.get_reference_lux()}")

        return lux

    def get_transverse_search_config(self) -> Dict[str, Any]:
        """
        Get the transverse search configuration.
        """
        awb = self.get_algorithm("awb")
        pos_range = awb["transverse_pos"] * 1
        neg_range = awb["transverse_neg"] * 1
        steps = floor((pos_range + neg_range) * 100 + 0.5) + 1
        steps = max(steps, 3)
        steps = min(steps, 12)
        steps *= 1
        config = {
            "pos_range": pos_range,
            "neg_range": neg_range,
            "transverse_steps": steps,
            "tangent_steps": 5,
            "tangent_range": 0.02
        }
        if "delta_limit" in awb:
            config["delta_limit"] = awb["delta_limit"]
        else:
            config["delta_limit"] = 0.15
        return config

    def get_transverse_search_range(self, temp: float, tangent_search: bool = False) -> float | None:
        """
        Get the colour values to check for searching perpendicular to the ct curve.
        """
        transverse = self.colour_temp_curve.transverse(temp)
        if transverse is None:
            return None

        if tangent_search:
            tangent = self.colour_temp_curve.tangent(temp)
            if tangent is None:
                return None

            tangent = tangent / np.linalg.norm(tangent)

        config = self.get_transverse_search_config()
        transverse_steps = np.linspace(-config["neg_range"], config["pos_range"], config["transverse_steps"])
        transverse_steps = np.expand_dims(transverse_steps, axis=1)
        transverse_steps = transverse_steps * transverse

        if tangent_search:
            tangent_steps = np.linspace(-config["tangent_range"], config["tangent_range"], config["tangent_steps"])
            tangent_steps = np.expand_dims(tangent_steps, axis=1)
            tangent_steps = tangent_steps * tangent

            steps = transverse_steps[None, :, :] + tangent_steps[:, None, :]
            steps = steps.reshape(-1, 2)

        else:
            steps = transverse_steps

        steps = steps + np.array(self.get_colour_values(temp))
        return steps