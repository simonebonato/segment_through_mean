import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from typing import List, Tuple, Optional


class SegThroughMeanExt:
    def __init__(self, data: np.ndarray):
        assert (
            data.shape[1] == 2 and len(data.shape) == 2
        ), "The shape of the input array is not correct. It should be (n, 2)"

        self.data = data
        self.mean_point = (np.mean(data[:, 0]), np.mean(data[:, 1]))

    def get_consecutive_segments(self) -> List[Tuple[float, float]]:
        """
        Method that takes in a list of 2D points representing the border of a shape and returns the list of consecutive segments.

        Returns
        :return: the list of consecutive segments
        """

        # create a df with the pairwise distances between the points
        distances_df = pd.DataFrame(pairwise_distances(self.data))

        # for each point, get the two closest points (excluding the point itself)
        # they will be the two points that form the segment of the contour
        consecutive_segments = []
        for point_idx in distances_df.index:
            ordered_row_index = distances_df.loc[point_idx].sort_values(ascending=True)

            origin_point = self.data[point_idx]
            first_neighbour = self.data[ordered_row_index.index[1]]

            second_neighbor_idx = 2
            second_neighbour = self.data[ordered_row_index.index[second_neighbor_idx]]

            consecutive_segments.append((origin_point, first_neighbour))

            # get the line perpendicular to the first line, and that passes through the origin point
            line = self.get_perpendicular_line(origin_point, first_neighbour)

            if line is None:
                continue

            # check if the second neighbour is on the same side of the line, else take the next one
            while (
                line(first_neighbour[0], first_neighbour[1])
                * line(second_neighbour[0], second_neighbour[1])
                > 0
            ):
                second_neighbor_idx += 1
                if second_neighbor_idx == len(ordered_row_index):
                    break
                second_neighbour = self.data[
                    ordered_row_index.index[second_neighbor_idx]
                ]

            consecutive_segments.append((origin_point, second_neighbour))

        return consecutive_segments

    @staticmethod
    def get_perpendicular_line(
        origin_point: Tuple[float, float], first_neighbour: Tuple[float, float]
    ):
        """
        Given the origin point and the first neighbour, return the function
        that describes the perpendicular line that passes through the origin point.
        If it exists, else return None.

        Parameters
        :param origin_point: the origin point
        :param first_neighbour: the first neighbour

        Return
        :return: the function that describes the perpendicular line, or None if it doesn't exist
        """
        m_num = first_neighbour[1] - origin_point[1]
        m_den = first_neighbour[0] - origin_point[0]

        # if the two points are the same, skip
        if m_den == 0 and m_num == 0:
            return None
        # if the two points are on the same vertical line, the perpendicular line is horizontal
        elif m_den == 0:
            return lambda x, y: origin_point[0] - x
        # if the two points are on the same horizontal line, the perpendicular line is vertical
        elif m_num == 0:
            return lambda x, y: origin_point[1] - y
        # else calculate the slope and intercept of the perpendicular line
        else:
            m = m_num / m_den
            m_perp = -1 / m
            q_perp = origin_point[1] - m_perp * origin_point[0]
            return lambda x, y: m_perp * x + q_perp - y

    @staticmethod
    def get_segment_through_mean(
        starting_point: Tuple[float, float],
        mean_point: Tuple[float, float],
        consecutive_segments: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Given a starting points on the border, the mean point of the shape and the consecutive segments
        of the border, return the list of points that form the segment that starts from the starting point,
        goes through the mean, and arrives on another point on the border of the shape.

        Parameters
        :param starting_point: the starting point of the segment
        :param mean_point: the mean point of the shape
        :param consecutive_segments: the list of consecutive segments

        Return
        :return: the list of points that form the segment
        """
        # line that passes through the first point and the mean point
        m = (mean_point[1] - starting_point[1]) / (mean_point[0] - starting_point[0])
        b = mean_point[1] - m * mean_point[0]

        intersection_points = [(starting_point[0], starting_point[1])]

        # for each segment, find the line that passes through it,
        # check if it intersects with the line that passes through the mean point and the starting point
        for segment in consecutive_segments:
            x1, y1 = segment[0]
            x2, y2 = segment[1]

            # check if the segment is vertical
            if x2 - x1 != 0:
                # calculate the slope and intercept of the line that passes through the segment
                m2 = (y2 - y1) / (x2 - x1)
                b2 = y1 - m2 * x1

                # calculate the intersection of the lines
                x = (b2 - b) / (m - m2)
                y = m * x + b

                # check if the intersection point is within the segment
                if (x1 <= x <= x2 or x2 <= x <= x1) and (
                    y1 <= y <= y2 or y2 <= y <= y1
                ):
                    intersection_points.append((x, y))

            else:
                # if the segment is vertical, the intersection point is the x of the segment
                x = x1
                y = m * x + b

                # check if the intersection point is within the segment
                if y1 <= y <= y2 or y2 <= y <= y1:
                    intersection_points.append((x, y))

        intersection_points = list(set(intersection_points))
        return intersection_points

    @staticmethod
    def process(
        input_list: List[Tuple[float, float]], threshold: float = 1e-1
    ) -> List[Tuple[float, float]]:
        """
        Takes as input a list representing the points that make a segment, and removes the points that are too close to each other.
        Necessary because sometimes the segments that are extracted in the previous stages contain multiple ploints that are too close to each other.

        Parameters
        :param input_list: the list of points that make a segment
        :param threshold: the threshold that defines how close two points can be to be considered the same point

        Return
        :return: the list of points that make a segment, without the points that are too close to each other

        """
        combos = itertools.combinations(input_list, 2)
        points_to_remove = [
            point2
            for point1, point2 in combos
            if abs(point1[0] - point2[0]) <= threshold
            and abs(point1[1] - point2[1]) <= threshold
        ]

        points_to_keep = [
            point for point in input_list if point not in points_to_remove
        ]

        return points_to_keep

    def get_all_segments_through_mean(
        self,
        optional_points: Optional[np.ndarray] = None,
    ) -> List[List[Tuple[float, float]]]:
        """
        Method that takes in a list of 2D points representing the border of a shape and returns the list of segments that start from a point on the border,
        pass through the mean point and end on another point on the border. Each segment is defined by the points at the extremes.

        Parameters
        :param optional_points: the list of points that make the border of the shape, if you don't want to use the points that were used to create the object

        Returns
        :return: the list of segments that pass through the mean point
        """

        # get all the consecutive segments of the border
        consecutive_segments = self.get_consecutive_segments()

        if optional_points is not None:
            points_to_use = optional_points
        else:
            points_to_use = self.data

        # save the mean point
        mean_point = (np.mean(points_to_use[:, 0]), np.mean(points_to_use[:, 1]))

        # for each point on the border, get the segment that passes through the mean point and ends on another point on the border
        segments_through_mean = []
        for point in points_to_use:
            segments_through_mean.append(
                self.get_segment_through_mean(point, mean_point, consecutive_segments)
            )

        # remove duplicates since sometimes the founds segments have the same points
        segments_through_mean = [
            self.process(segment)
            for segment in segments_through_mean
            if len(segment) > 1
        ]

        # delete all the segments with only one point
        segments_through_mean = [
            segment for segment in segments_through_mean if len(segment) > 1
        ]

        return segments_through_mean

    @staticmethod
    def get_max_segment(
        segments: List[List[Tuple[float, float]]]
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Method that takes in a list of segments and returns the longest one and its length.

        Parameters
        :param segments: the list of segments

        Returns
        :return: the longest segment and its length
        """

        # set max length to 0 and max segment to None
        max_length = 0.0
        max_segment = None

        # for each segment, calculate its length and check if it is the longest
        for segment in segments:
            dist = np.linalg.norm(np.array(segment[0]) - np.array(segment[1]))

            if dist > max_length:
                max_length = dist
                max_segment = segment

        return (max_segment, max_length)

    def add_intermediate_points(
        self, times: int = 2, optional_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Method that takes in a list of 2D points representing the border of a shape
        and return the same points, plus the midpoints of the segments that connect them.
        It can be repeated multiple times to obtain a more precise measure of the max segment.

        Parameters
        :param times: the number of times to repeat the operation
        :param optional_points: the list of points to use instead of the ones of the object

        Returns
        :return: the list of points, plus the midpoints of the segments that connect them
        """
        if optional_points is not None:
            points_to_use = optional_points
        else:
            points_to_use = self.data.copy()

        for _ in range(times):
            consecutive_segments = self.get_consecutive_segments()

            for segment in consecutive_segments:
                # compute the midpoint of the segment
                midpoint = (segment[0] + segment[1]) / 2
                # add the midpoint to the list of points
                points_to_use = np.vstack((points_to_use, midpoint))

        return points_to_use
