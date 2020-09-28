%% Finite Impulse Response
clear
clc

format long;

ecg = load('enel420_grp_1.txt', '-ascii');
N = length(ecg);
fs = 1024;
t = (1:1:length(ecg))/fs;

% Plot the Unfiltedered ECG Signal
figure(1);
spectrum = abs(fft(ecg));
nfreq = ((spectrum / N) * fs);
plot(nfreq);
grid minor;
title("Unfiltered Frequency Spectrum");
xlabel("Frequency (Hz)");
ylabel("Voltage (uV))");

% Plot the Unfiltedered ECG Spectrum
figure(2);
plot(20*log(spectrum));
grid on;
title("Unfiltered Frequency Spectrum");
xlabel("Frequency (Hz)");
ylabel("Magnitude (dB)");

% The undesired frequencies and desired bandwidth of
freq1 = 31.456;
freq2 = 74.36;
BW = 5;

deg1 = 2*pi * (freq1 / fs);
deg2 = 2*pi * (freq2 / fs);
r = 1 - (BW / fs) * pi;

% Assign the coefficients for first and second filters
a = 1 * 1;
b = (1*-exp(-deg1*1i)) + (1*-exp(deg1*1i));
c = (1*-exp(-deg1*1i)) * (1*-exp(deg1*1i));

d = 1 * 1;
e = (-r*exp(-deg1*1i)) + (-r*exp(deg1*1i));
f = (-r*exp(-deg1*1i)) * (-r*exp(deg1*1i));

g = 1 * 1;
h = (-1*exp(-deg2*1i)) + (-1*exp(deg2*1i));
ii = (-1*exp(-deg2*1i)) * (-1*exp(deg2*1i));

j = 1 * 1;
k = (-r*exp(-deg2*1i)) + (-r*exp(deg2*1i));
l = -r*exp(-deg2*1i) * -r*exp(deg2*1i);

% Calculte the gain of the overall transfer function
Wf = 2 * pi *10;
ND_array = [exp(0), exp(1i*Wf), exp(-2*Wf)];
H_Z1_dot = dot(ND_array,[a b c]);
H_Z2_dot = dot(ND_array, [d e f]);
Gain = abs(H_Z2_dot / H_Z1_dot);

% convlute the the de/numerator of the first transfer function with de/numerator
% of the second funcion
NUM_Z = conv([a b c], [g h ii]);
DEN_Z = conv([d e f], [j k l]);


[H, F] = freqz(Gain*NUM_Z, DEN_Z, N, fs);
figure(3);
plot(F, abs(H));
title("Infinite Impulse Response");
xlabel("Frequency (Hz)");
ylabel("Magnitude");
grid on;

figure(4);
subplot(2, 1, 1);
plot(t, ecg);
grid on;
title("Unfiltered ECG Signal");
xlabel("Time (sec)");
ylabel("Voltage (uV)");


y = filter(NUM_Z, DEN_Z, ecg);
subplot(2, 1, 2);
plot(t, y);
grid on;
title("IIR Filtered ECG Signal");
xlabel("Time (sec)");
ylabel("Voltage (uV)");

% Calculate the variance of the unfiltered and the filtered signals and
% find the noise by taking the difference
unfiltered_pwr = var(ecg);
filtered_pwr = var(y);
noise_in_ECG = abs(filtered_pwr - unfiltered_pwr);