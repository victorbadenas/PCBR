import numpy as np

from constraints import Constraints

class UserRequest:
    profile_format = ["Experience", "WFH", "Primary use", "Budget", 
        "Replace (1-most frequent; 4-least frequent)", 
        "Office", "Photoshop", "VideoChat", "ML", "Compilers", 
        "HighPerformanceGames", "LowPerformanceGames"]

    def __init__(self, profile_str, pref_str, constraints_str, scalers, feature_relevance_matrix):
        self.profile = self._process_profile(profile_str, scalers)
        self.preferences = self._process_preferences(pref_str, feature_relevance_matrix, scalers)
        self.constraints = self._process_constraints(constraints_str, scalers)

    def _process_profile(self, profile_str:str, scalers:dict=None) -> np.ndarray:
        # Input format: Experience, WFH, Primary Use, Budget, Replace, Office, Photoshop, VideoChat, ML, Compilers, HighPerformanceGames, LowPerformanceGames
        # Input example (exactly matches case 2): '2, 1, Programming, 1, 3, 1, 0, 0, 0, 1, 0, 0'
        # Output format: numpy array suitable to look up nearest case in case library (Note: you will need to
        #                apply pre-processing to obtain the correct output format)
        # Example output: [[0.25       1.         0.6        0.         0.66666667 1.
        #                   0.         0.         0.         1.         0.         0.        ]]
        profile = profile_str.split(',')
        if scalers is None:
            return np.array(list(map(float, profile)))

        for i, (column, value) in enumerate(zip(self.profile_format, profile)):
            column, value = column.strip(), value.strip()
            mapper = scalers[column]
            if 'map' in mapper:
                # categorical require a mapping
                value = mapper['map'][value]
            else:
                # else convert to float
                value = float(value)

            if mapper['log2']:
                value = np.log2(value+1)

            profile[i] = mapper['scaler'].transform([[value]])[0,0]
        return np.array(profile).reshape(1, -1)

    def _process_preferences(self, pref_str:str, feature_relevance_matrix:np.ndarray, scalers:dict=None) -> np.ndarray:
        # Input format: Preferences matrix survey answers (string). Importance on scale of 1-5, where 1 is least
        #               and 5 is most. Categories are: budget, performance, multi-tasking, gaming,
        #               streaming videos, editing videos/photos/music, fast startup/shutdown, video chat
        # Input example: '5, 2, 3, 1, 2, 1, 3, 4, 1, 0, 1, 0, 0'
        # Output format: numpy array of answers
        # Example output: [5 2 3 1 2 1 3 4 1 0 1 0 0]
        preferences_arr = list(map(int, pref_str.split(',')))
        preferences_scaled = (np.array(preferences_arr).astype(np.float) - 1) / 4
        feature_relevance = preferences_scaled@feature_relevance_matrix
        feature_relevance_scaled = feature_relevance / 2 + .5
        return feature_relevance_scaled


    def _process_constraints(self, constraints_str:str, scalers:dict=None) -> Constraints:
        # Input format: string with multiple comma-separated key: value pairs of constraints. 
        # Input example: 'cpu_brand: Intel, gpu_brand: PreferNVIDIA, max_budget: 1000'
        # Output format: Constraints object
        # Kevin: This part will be processed after the weighted kNN so as to try to solve the different constraints
        #        by giving different options to the user via UI.
        constraints_dict = dict()
        for constraint in constraints_str.split(","):
            k, v = constraint.split(':')
            constraints_dict[k.strip()] = v.strip()
        return Constraints(constraints_dict)


