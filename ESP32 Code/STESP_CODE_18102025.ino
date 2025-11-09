#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <sensor_msgs/msg/laser_scan.h>
#include <geometry_msgs/msg/twist.h>
#include <sensor_msgs/msg/imu.h>
#include <sensor_msgs/msg/magnetic_field.h>
#include <nav_msgs/msg/odometry.h>
#include <WiFi.h>
#include <RPLidar.h>
#include <math.h>
#include <SPI.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <rmw_microros/rmw_microros.h>

//Pin intiailization
#define RPLIDAR_MOTOR 14 //LIDAR pins
#define RPLIDAR_RX    16
#define RPLIDAR_TX    17

#define LED_PIN       13 //Error loop LED

#define STM32_CS 5     //SPI pins
#define SPI_MOSI 23    
#define SPI_MISO 19
#define SPI_SCK 18

//WiFi and micro-ROS agent
char WIFI_SSID[]     = "Dieu l'a fait";
char WIFI_PASSWORD[] = "123456713";
char AGENT_IP[]      = "10.19.3.71";
uint16_t AGENT_PORT  = 8888;

//ROS entities
RPLidar lidar;
rcl_publisher_t imu_publisher; //imu ros publisher
rcl_publisher_t lidar_publisher; //lidar ros publisher
rcl_subscription_t cmd; // twist msg subscriber
rcl_publisher_t odom_publisher; //odometry ros publisher
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer1; //timer 1 initiailising
rcl_timer_t timer2; //timer 2 initialising

//micto ROS messages
sensor_msgs__msg__LaserScan scan_msg;
geometry_msgs__msg__Twist cmdvel_msg;
sensor_msgs__msg__Imu imu_msg;
nav_msgs__msg__Odometry odom_msg;


//SPI
SPIClass *vspi = nullptr;

uint8_t txbuf[32]; //SPI buffers
uint8_t rxbuf[32];

int16_t imu_data[6] = {0};   //IMU data (ax,ay,az,gx,gy,gz)
float odom_data[5] = {0.0};  //Odometry data
volatile bool new_data_available = false;

int16_t ax, ay, az, gx, gy, gz; //IMU values
float accX, accY, accZ, gyroX, gyroY, gyroZ;

float twistmsg[6] = {0.0}; //Twist message

const int timeout_ms = 1000; 


//mnicro-ROS error handling
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();} }
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){} }

void error_loop() { //error loop for micro-ROS errors
  while (1) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

//SPI Task
void spiImuTask(void *pvParameters) {

  for (;;) {
    
    memcpy(txbuf, twistmsg, 24); //copy latest twist msg to transfer buf
    memset(txbuf + 24, 0, 8); //fill with zeros
    vspi->beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0)); //1MHz SPI with mode 0
    digitalWrite(STM32_CS, LOW);

    vspi->transferBytes(txbuf, rxbuf, 32); //send and recieve data

    digitalWrite(STM32_CS, HIGH);
    vspi->endTransaction();

    memcpy(imu_data, rxbuf, 12); //copy first 12 bytes to imu data
    memcpy(odom_data, rxbuf + 12, 20); // copy the remaining 20 bytes to odom data
    new_data_available = true; //set flag for new data
    
  }
}

//Accelerometer calculations
void calculateAccGyroTask(void*pvParamters){

  for(;;){
    if (new_data_available) { // assign new IMU values recieved
      noInterrupts();
      int16_t ax = imu_data[0];
      int16_t ay = imu_data[1];
      int16_t az = imu_data[2];
      int16_t gx = imu_data[3];
      int16_t gy = imu_data[4];
      int16_t gz = imu_data[5];
      new_data_available = false;
      interrupts();

      accX = ax / 16384.0 * 9.81; //at +-2g 16384 LSB/g as per data sheet
      accY = ay / 16384.0 * 9.81; //accelaration interms of ms-2
      accZ = az / 16384.0 * 9.81;

      gyroX = gx / 131.0 * (M_PI / 180.0);  //angular velocity rad/s
      gyroY = gy / 131.0 * (M_PI / 180.0);  //at +-250 deg/s 131 LSB/(deg/s)
      gyroZ = gz / 131.0 * (M_PI / 180.0);
    }
  }
}

#define FRONT_SCAN_POINTS 181 // -90 to +90, 181 to support ROS2 SLAM toolbox as min is 181 points

static float front_ranges[FRONT_SCAN_POINTS];
static float front_intensities[FRONT_SCAN_POINTS];
static bool initialized = false;

void LidarTask(void *pvParameters){
  for(;;){
    if (IS_OK(lidar.waitPoint())) { //if lidar have data
      float angle = lidar.getCurrentPoint().angle; // degrees
      float distance = lidar.getCurrentPoint().distance / 1000.0f; // mm to m
      byte quality = lidar.getCurrentPoint().quality; //get intensity

      //Keep only front sector -90 - +90
      if (angle >= 90.0 && angle < 270.0) {
        int index = (int)(angle - 90.0); // map 90–270 as index 0–180
        if (index >= 0 && index < FRONT_SCAN_POINTS) {
            front_ranges[index] = distance; //assign ranegs
            front_intensities[index] = (float)quality; //assign intensities
        }
      }

      //When a full scan is completed
      if (lidar.getCurrentPoint().startBit) {
        if (!initialized) {
          //First scan initialization
          for (int i = 0; i < FRONT_SCAN_POINTS; i++) {
            front_ranges[i] = NAN;
            front_intensities[i] = NAN;
          }
          initialized = true;
        }
        scan_msg.angle_min = -M_PI / 2;   // -90
        scan_msg.angle_max = M_PI / 2;    // +90
        scan_msg.angle_increment = M_PI / 180; // pi / 180, increament of 1 degree.
        scan_msg.time_increment = 0.0;
        scan_msg.scan_time = 0.1;
        scan_msg.range_min = 0.15; //min range
        scan_msg.range_max = 12.0; // max range

        for (int i = 0; i < FRONT_SCAN_POINTS; i++) {
          int reversed_index = FRONT_SCAN_POINTS - 1 - i; //flip scan to support ROS2 CCW direction
          scan_msg.ranges.data[i] = front_ranges[reversed_index];
          scan_msg.intensities.data[i] = front_intensities[reversed_index];
        }
      }
    }
  }   
}

//micro ROS timer callbacks
void timer_callback1(rcl_timer_t *timer, int64_t last_call_time) { //LIDAR timeer
  RCLC_UNUSED(timer);
  RCLC_UNUSED(last_call_time);

  uint64_t ms = rmw_uros_epoch_millis(); //get synced time fr time stamp
  scan_msg.header.stamp.sec = ms / 1000;
  scan_msg.header.stamp.nanosec = (ms % 1000) * 1000000;

  // Publish scan
  RCSOFTCHECK(rcl_publish(&lidar_publisher, &scan_msg, NULL));
    
  
}

void timer_callback2(rcl_timer_t * timer, int64_t last_call_time)
{
  RCLC_UNUSED(timer);
  RCLC_UNUSED(last_call_time);
  uint64_t ms = rmw_uros_epoch_millis(); //Get synced time for time stamp

  //IMU message
  imu_msg.header.stamp.sec = ms / 1000;
  imu_msg.header.stamp.nanosec = (ms % 1000) * 1000000;

  //Fill IMU message from current sensor data
  imu_msg.linear_acceleration.x = accX;
  imu_msg.linear_acceleration.y = accY;
  imu_msg.linear_acceleration.z = accZ;

  imu_msg.angular_velocity.x = gyroX;
  imu_msg.angular_velocity.y = gyroY;
  imu_msg.angular_velocity.z = gyroZ;

  imu_msg.orientation.x = 0.0;
  imu_msg.orientation.y = 0.0;
  imu_msg.orientation.z = 0.0;
  imu_msg.orientation.w = 1.0;

  // Publish imu messsage
  RCSOFTCHECK(rcl_publish(&imu_publisher, &imu_msg, NULL));

  //Odometry message
  float x = odom_data[0];
  float y = odom_data[1];
  float yaw = odom_data[2];
  float linear_vel = odom_data[3];
  float angular_vel = odom_data[4];

  odom_msg.header.stamp.sec = imu_msg.header.stamp.sec;
  odom_msg.header.stamp.nanosec = imu_msg.header.stamp.nanosec;
  odom_msg.header.frame_id.data = (char *)"odom";
  odom_msg.child_frame_id.data = (char *)"base_link"; //required for proper transform in ROS

  // Pose message
  odom_msg.pose.pose.position.x = x;
  odom_msg.pose.pose.position.y = y;
  odom_msg.pose.pose.position.z = 0.0;

  odom_msg.pose.pose.orientation.x = 0.0;
  odom_msg.pose.pose.orientation.y = 0.0;
  odom_msg.pose.pose.orientation.z = sin(yaw / 2.0);
  odom_msg.pose.pose.orientation.w = cos(yaw / 2.0);

  // Twist message
  odom_msg.twist.twist.linear.x = linear_vel;
  odom_msg.twist.twist.angular.z = angular_vel;

  // Covariances 
  for (int i = 0; i < 36; i++) {
    odom_msg.pose.covariance[i] = 0.0;
    odom_msg.twist.covariance[i] = 0.0;
  }

  //Publsih odometry message
  RCSOFTCHECK(rcl_publish(&odom_publisher, &odom_msg, NULL));

}


//Subscription callback
void subscription_callback(const void *msgin) { //obtain twist message from host
  twistmsg[0] = cmdvel_msg.linear.x;
  twistmsg[1] = cmdvel_msg.linear.y;
  twistmsg[2] = cmdvel_msg.linear.z;
  twistmsg[3] = cmdvel_msg.angular.x;
  twistmsg[4] = cmdvel_msg.angular.y;
  twistmsg[5] = cmdvel_msg.angular.z;
}

//Setup code
void setup() {
  Serial.begin(115200); //serial coms
  delay(1000);

  pinMode(LED_PIN, OUTPUT);

  set_microros_wifi_transports(WIFI_SSID, WIFI_PASSWORD, AGENT_IP, AGENT_PORT); //set up microros over udp
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  pinMode(RPLIDAR_MOTOR, OUTPUT);
  digitalWrite(RPLIDAR_MOTOR, HIGH);

  Serial2.begin(115200, SERIAL_8N1, RPLIDAR_RX, RPLIDAR_TX); //set up lidatr uart
  lidar.begin(Serial2);
  lidar.startScan();

  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "micro_ros_node", "", &support));

  RCCHECK(rclc_publisher_init_default( //lidar publisher
    &lidar_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, LaserScan),
    "scan"
  ));

  RCCHECK(rclc_publisher_init_default(//odometry publisher
  &odom_publisher,
  &node,
  ROSIDL_GET_MSG_TYPE_SUPPORT(nav_msgs, msg, Odometry),
  "odom"));

  RCCHECK(rclc_publisher_init_default(//imu publisher
    &imu_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
    "imu_data"));

  Serial.print("Publisher ready");

  RCCHECK(rclc_subscription_init_default(//cmd vel subscriber
    &cmd,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "cmd_vel"));

  Serial.print("Subscriber ready");

  //lidar message
  scan_msg.ranges.capacity = FRONT_SCAN_POINTS;
  scan_msg.ranges.size = FRONT_SCAN_POINTS;
  scan_msg.ranges.data = (float*)malloc(scan_msg.ranges.capacity * sizeof(float));

  scan_msg.intensities.capacity = FRONT_SCAN_POINTS;
  scan_msg.intensities.size = FRONT_SCAN_POINTS;
  scan_msg.intensities.data = (float*)malloc(scan_msg.intensities.capacity * sizeof(float));

  const char *frame_id = "laser_frame";
  scan_msg.header.frame_id.data = (char*)malloc(strlen(frame_id) + 1);
  strcpy(scan_msg.header.frame_id.data, frame_id);
  scan_msg.header.frame_id.capacity = strlen(frame_id) + 1;
  scan_msg.header.frame_id.size = strlen(frame_id);

  RCCHECK(rclc_timer_init_default2( //timer for lidar
    &timer1,
    &support,
    RCL_MS_TO_NS(10), //100Hz
    timer_callback1,
    true
  ));

  RCCHECK(rclc_timer_init_default2( //timer for imu and odom
    &timer2,
    &support,
    RCL_MS_TO_NS(10),   // 100 Hz
    timer_callback2,
    true));
  


  RCCHECK(rclc_executor_init(&executor, &support.context, 3, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &cmd, &cmdvel_msg, &subscription_callback, ON_NEW_DATA)); //add subcriber and timer to microros excutor
  RCCHECK(rclc_executor_add_timer(&executor, &timer1));
  RCCHECK(rclc_executor_add_timer(&executor, &timer2));


  // initialize imu_msg fields that won't change
  imu_msg.header.frame_id.data = (char *)"imu_link";
  imu_msg.header.frame_id.size = strlen("imu_link");
  imu_msg.header.frame_id.capacity = 10;
  //orientation covariance
  for (int i = 0; i < 9; i++)
    imu_msg.orientation_covariance[i] = 0.0;  // clear all
  imu_msg.orientation_covariance[0] = 0.05;   // roll variance
  imu_msg.orientation_covariance[4] = 0.05;   // pitch variance
  imu_msg.orientation_covariance[8] = 0.05;   // yaw variance

  //angular velocity covariance
  for (int i = 0; i < 9; i++)
    imu_msg.angular_velocity_covariance[i] = 0.0;
  imu_msg.angular_velocity_covariance[0] = 0.02;  // ωx
  imu_msg.angular_velocity_covariance[4] = 0.02;  // ωy
  imu_msg.angular_velocity_covariance[8] = 0.02;  // ωz

  //linear acceleration covariance
  for (int i = 0; i < 9; i++)
    imu_msg.linear_acceleration_covariance[i] = 0.0;
  imu_msg.linear_acceleration_covariance[0] = 0.04;  // ax
  imu_msg.linear_acceleration_covariance[4] = 0.04;  // ay
  imu_msg.linear_acceleration_covariance[8] = 0.04;  // az

  //initialize odom feilds that wont change
  odom_msg.header.frame_id.data = (char *)"odom";
  odom_msg.child_frame_id.data = (char *)"base_link";
  odom_msg.header.frame_id.size = strlen("odom");
  odom_msg.child_frame_id.size = strlen("base_link");
  odom_msg.header.frame_id.capacity = 10;
  odom_msg.child_frame_id.capacity = 10;


  pinMode(STM32_CS, OUTPUT);
  digitalWrite(STM32_CS, HIGH);

  vspi = new SPIClass(VSPI); //initialise spi
  vspi->begin(SPI_SCK, SPI_MISO, SPI_MOSI, STM32_CS);

  xTaskCreatePinnedToCore( //FreeRTOS task for SPI task
    spiImuTask,
    "SPI_IMU_Task",
    1024,
    NULL,
    1,
    NULL,
    0
  );

  xTaskCreatePinnedToCore( //FreeRTOS task for calculating IMU msg values task
    calculateAccGyroTask, 
    "IMU_Value_Task", 
    1024, 
    NULL, 
    1, 
    NULL, 
    0
  );

  xTaskCreatePinnedToCore(//FreeRTOS task for Lidar data acquisiion
    LidarTask, 
    "LIDAR_Task", 
    1024, 
    NULL, 
    1, 
    NULL, 
    1
  );



}

//Loop 
void loop() {
  RCCHECK(rmw_uros_sync_session(timeout_ms)); //sync time
  static unsigned long lastPrintTime = 0;
  unsigned long now = millis();

  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1)); //run micrros excutor

  //print for debugging
  if (now - lastPrintTime >= 100) { 
    lastPrintTime = now;

    Serial.print("Accel X: "); Serial.print(accX); Serial.print(" ");
    Serial.print("Accel Y: "); Serial.print(accY); Serial.print(" ");
    Serial.print("Accel Z: "); Serial.print(accZ); Serial.print(" ");
    Serial.print("Gyro X: "); Serial.print(gyroX); Serial.print(" ");
    Serial.print("Gyro Y: "); Serial.print(gyroY); Serial.print(" ");
    Serial.print("Gyro Z: "); Serial.println(gyroZ);

    Serial.print("linear.x: "); Serial.print(cmdvel_msg.linear.x);
    Serial.print(", linear.y: "); Serial.print(cmdvel_msg.linear.y);
    Serial.print(", linear.z: "); Serial.print(cmdvel_msg.linear.z);
    Serial.print(" | angular.x: "); Serial.print(cmdvel_msg.angular.x);
    Serial.print(", angular.y: "); Serial.print(cmdvel_msg.angular.y);
    Serial.print(", angular.z: "); Serial.println(cmdvel_msg.angular.z);

    Serial.print("Odom X: "); Serial.print(odom_data[0]);
    Serial.print(" Y: "); Serial.print(odom_data[1]);
    Serial.print(" Yaw: "); Serial.print(odom_data[2]);
    Serial.print(" LinVel: "); Serial.print(odom_data[3]);
    Serial.print(" AngVel: "); Serial.println(odom_data[4]);
  }
}
