%% Dynamic Positioning Model for a Ship in 2D

% Clear workspace and command window
clear; clc;

%% Ship Parameters
m = 1000;   % Mass of the ship (kg)
I = 500;    % Yaw moment of inertia (kg*m^2)

%% State-Space Formulation
% States:
% x(1) = x-position, x(2) = y-position, x(3) = heading (psi)
% x(4) = x-velocity, x(5) = y-velocity, x(6) = yaw rate (psi_dot)
%
% Dynamics:
%   dx/dt = x_dot
%   dy/dt = y_dot
%   dpsi/dt = psi_dot
%   d(x_dot)/dt = Fx/m
%   d(y_dot)/dt = Fy/m
%   d(psi_dot)/dt = Mz/I

A = [0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0];

B = [0    0    0;
     0    0    0;
     0    0    0;
     1/m  0    0;
     0    1/m  0;
     0    0    1/I];

% For simplicity, assume the output y equals the full state vector:
C = eye(6);
D = zeros(6,3);

% Create the state-space system object
sys = ss(A, B, C, D);

%% Simulation Setup
% Define simulation time
tspan = 0:0.1:20;  % simulation from 0 to 20 seconds

% Initial state [x; y; psi; x_dot; y_dot; psi_dot]
initial_state = [0; 0; 0; 0; 0; 0];

%% Define Control Inputs
% Here we assume a constant force in the x-direction and a moment for turning.
% In a real DP (dynamic positioning) system, these would be computed by a controller.
u = zeros(3, length(tspan));
u(1, :) = 100;  % Constant force Fx (N)
u(3, :) = 10;   % Constant moment Mz (N*m)

%% Simulate the System
% lsim simulates the time response of the state-space system
[y, t, x] = lsim(sys, u', tspan, initial_state);

%% Plotting the Results
figure;

% Plot positions
subplot(3,1,1);
plot(t, x(:,1), 'b', t, x(:,2), 'r');
xlabel('Time (s)');
ylabel('Position (m)');
legend('x-position','y-position');
title('Ship Position');

% Plot velocities
subplot(3,1,2);
plot(t, x(:,4), 'b', t, x(:,5), 'r');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('x-velocity','y-velocity');
title('Ship Velocities');

% Plot heading (convert radian to degree)
subplot(3,1,3);
plot(t, x(:,3)*180/pi, 'k');
xlabel('Time (s)');
ylabel('Heading (deg)');
title('Ship Heading');
