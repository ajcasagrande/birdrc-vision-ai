# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

import numpy as np

from .basetrack import TrackState
from .bot_sort import BOTSORT, BOTrack
from .utils.matching import linear_assignment


class EnhancedBotSort(BOTSORT):
    def __init__(self, args, frame_rate=30, max_match_distance=50):
        """
        Initialize EnhancedBotSort tracker with additional handling for lost tracks.

        Args:
            args (Namespace): Command-line arguments.
            frame_rate (int): Frame rate of the video sequence.
            max_match_distance (float): Maximum distance for matching new detections to lost tracks.
        """
        super().__init__(args, frame_rate)
        self.max_match_distance = max_match_distance
        self.lost_tracks = []
        self.active_tracks = []
        self.removed_tracks = []

    def update(self, results, img=None):
        """
        Updates the tracker with new detections and matches to lost tracks if applicable.

        Args:
            results: Detected objects from YOLO or another source.
            img (Optional): Image for feature updates.

        Returns:
            np.ndarray: Tracked objects with updated states.
        """
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        scores = results.conf
        bboxes = results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores >= self.args.track_high_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cls_keep = cls[remain_inds]

        detections = self.init_botrack(dets, scores_keep, cls_keep, img)

        # Match new detections to lost tracks
        if len(self.lost_tracks) > 0 and len(detections) > 0:
            lost_track_bboxes = [track.xywh for track in self.lost_tracks]
            detection_bboxes = [det.xywh for det in detections]
            distances = np.linalg.norm(
                np.array(lost_track_bboxes)[:, None, :2] - np.array(detection_bboxes)[None, :, :2], axis=-1
            )
            matches, u_lost, u_detections = self.match_lost_to_detections(distances)

            # Reassign lost tracks based on matches
            for i, j in matches:
                lost_track = self.lost_tracks[i]
                det = detections[j]
                lost_track.re_activate(det, self.frame_id)
                refind_tracks.append(lost_track)

            # Update unmatched detections and lost tracks
            self.lost_tracks = [self.lost_tracks[i] for i in u_lost]
            detections = [detections[j] for j in u_detections]

        # Match active tracks to remaining detections
        track_pool = self.joint_tracks(self.active_tracks, self.lost_tracks)
        self.multi_predict(track_pool)

        dists = self.get_dists(track_pool, detections)
        matches, u_track, u_detections = self.match_active_tracks(dists)

        for i, j in matches:
            track = track_pool[i]
            det = detections[j]
            track.update(det, self.frame_id)
            activated_tracks.append(track)

        for it in u_track:
            track = track_pool[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        # Activate unmatched detections
        for idx in u_detections:
            det = detections[idx]
            det.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(det)

        # Update track states
        self.active_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        self.active_tracks = self.joint_tracks(self.active_tracks, activated_tracks)
        self.lost_tracks = self.sub_tracks(self.lost_tracks, self.active_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.removed_tracks.extend(removed_tracks)

        return np.asarray([t.result for t in self.active_tracks if t.is_activated], dtype=np.float32)

    def init_botrack(self, dets, scores, cls, img=None):
        """
        Initializes tracks using BOTrack.

        Args:
            dets: Detection boxes.
            scores: Confidence scores.
            cls: Class IDs.

        Returns:
            List[BOTrack]: List of initialized tracks.
        """
        return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []

    def match_lost_to_detections(self, distances):
        """
        Matches lost tracks to new detections within a specified range.

        Args:
            distances (np.ndarray): Distance matrix.

        Returns:
            Tuple: Matches, unmatched lost track indices, unmatched detection indices.
        """
        return linear_assignment(distances, self.max_match_distance)

    def match_active_tracks(self, distances):
        """
        Matches active tracks to detections.

        Args:
            distances (np.ndarray): Distance matrix.

        Returns:
            Tuple: Matches, unmatched active track indices, unmatched detection indices.
        """
        return linear_assignment(distances, self.args.match_thresh)

    @staticmethod
    def joint_tracks(tlista, tlistb):
        """
        Combines two lists of tracks into a single list, avoiding duplicates.

        Args:
            tlista: First list of tracks.
            tlistb: Second list of tracks.

        Returns:
            List: Combined list of tracks.
        """
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_tracks(tlista, tlistb):
        """
        Subtracts tracks in tlistb from tlista.

        Args:
            tlista: List of tracks.
            tlistb: Tracks to subtract.

        Returns:
            List: Filtered tracks.
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]
