# import numpy as np
#
# """
# X_train =
#     [
#         [1000, 21, 2],
#         [2000, 26, 4],
#         [300, 19, 1],
#     ]
#
# Dimensions are salary age and experience
#
# y_train =
#         [
#           "under_paid",
#           "over_paid",
#           "enough_paid",
#
#         ]
#
#
# Based on information gain we found that
#     best_feature = 1
#     best_thresh = 21
#
#
# best_feature = 1 = age = X_train[:,best_feature] = [21, 26, 19]
# predicate is <= 21
#
# Row 0: 21 <= 21  ✅
# Row 1: 26 <= 21  ❌
# Row 2: 19 <= 21  ✅
# """
#
# X_train = np.array([
#         [1000, 21, 2],
#         [2000, 26, 4],
#         [300, 19, 1],
#         [1500, 21, 2]])
#
#
# # In ML, the labels are converted to numerical values before training
# """
# "under_paid", = 0
# "over_paid", = 1
# "enough_paid", = 2
# """
#
#
# y_train = np.array([
#            0,
#            1,
#            2,
#            0,])
#
#
#
# best_feature_index = 1
# best_threshold = 21
#
#
# X_column = X_train[:,best_feature_index]
# left_row_indices = np.argwhere(X_column <= best_threshold).flatten()
# right_row_indices = np.argwhere(X_column > best_threshold). flatten()
# # print(left_row_indices)
# # print(right_row_indices)
# # print(y_train[left_row_indices])
#
# # index == numerical label and the value is the frequency
# binCount = np.bincount(y_train)
# print("bunCount = ", binCount)
# ps = binCount / len(y_train)
# print(ps)
#
