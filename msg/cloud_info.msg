# Cloud Info
Header header 

# 在进行点云去畸变时，把range数据展开成一维向量
# ring代表第几条激光线数，比如16线的激光雷达有16个ring
# start_ring_index记录第一个ring在一维向量中的起始索引
# end_ring_index记录第一个ring在一维向量中的结束索引
int32[] startRingIndex
int32[] endRingIndex

# 记录一维的Range数据中每一个点在原始range图片中属于那一个列
int32[]  pointColInd # point column index in range image
# 所有点的range数值，展开为一维数组
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
# 从IMU原始数据获取的roll、pitch、yaw角初值
# 当上面的imu_avaliable为true，odom_available为false的时候，会被用来
# 当作这一帧点云的初值
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
# 从imuPreintegration，IMU预积分模块获取的与这一帧雷达最接近的初始位姿
# 当上面的odom_available标签被置为true的时候会被使用
# 优先级比imu_xxx_init高
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
# 去畸变后的原始点云
# 在经过imageProjection模块后，这个数据字段被填充。在经过特征提取之后这个字段被清空
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
# featureExtraction模块提取出的点云边缘点
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
# featureExtraction模块提取出的点云平面点
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

# 3rd party messages
sensor_msgs/PointCloud2 key_frame_cloud
sensor_msgs/PointCloud2 key_frame_color
sensor_msgs/PointCloud2 key_frame_poses
sensor_msgs/PointCloud2 key_frame_map
