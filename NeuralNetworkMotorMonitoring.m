clear all, close all, clc

% Electric motor bearing monitoring utilizing neural networks

% Author: Tuomas Du-Ikonen
% Date: 29.6.2021

%% LOAD DATA
load('datacombined.mat')

% Lets get some variables needed in this program.
Lsgn = size(fltvector,1); %Signal lenght. Returns only number of rows.

%% DIVIDE DATA TO SAMPLES

% Parameters
% I don't know number of samples and features that would NN to work fast
% enough and with sufficient accuracy. Let's make them as parameters so we
% can adjust them later easily.
samples = 100; 
features = 20;
% This is the highest frequency of motor sound that we're interested.
% Later we will leave out higher frequences. We know this 
% because of testing and it just makes plotting easier.
maxHz = 8000;
ylimmax = 0.008; % Max y-axis value to plot.

testsample = 1; % Index of sample array to plot to demonstrate it.


% Lets divide data into samples.
samplelenght = floor(Lsgn/samples); %Round down sample lenght to make calculations easier. Rest of the data we're going to just discard.

faultytrainingset = zeros(samplelenght, samples);
normaltrainingset = zeros(samplelenght, samples);
jj = 1;
for ii = 1:samples
    faultytrainingset(:,ii) = fltvector(jj:(jj+samplelenght-1),1:1);
    normaltrainingset(:,ii) = nrmlvector(jj:(jj+samplelenght-1),1:1);
    jj = jj + samplelenght;
end
%% PLOT SIGNALS IN TIME DOMAIN

f = figure;
f.Position = [200 200 1500 600];

subplot(2,4,1);
plot(fltvector, 'b');
ylim([-0.1,0.1]);
xlim([1,500000]);
title('Faulty motor time domain');
xlabel('Samples')

subplot(2,4,5);
plot(nrmlvector, 'r');
ylim([-0.1,0.1]);
xlim([1,500000]);
title('Normal motor time domain');
xlabel('Samples')
%% FFT TRANSFORMATION

% Array of faulty and normal training set for to be send for training.
faultyts = zeros(samplelenght/2+1, samples);
normalts = zeros(samplelenght/2+1, samples);

for ii = 1:samples
    FLTsample = fft(faultytrainingset(:,ii));   %FFT transformation for faulty signal sample.
    P2flt2 = abs(FLTsample/samplelenght);
    P1flt2 = P2flt2(1:samplelenght/2+1);
    P1flt2(2:end-1) = 2*P1flt2(2:end-1);
    faultyts(:,ii) = P1flt2;
    
    NRMLsample = fft(normaltrainingset(:,ii));   %FFT transformation for normal signal sample.
    P2flt2 = abs(NRMLsample/samplelenght);
    P1flt2 = P2flt2(1:samplelenght/2+1);
    P1flt2(2:end-1) = 2*P1flt2(2:end-1);
    normalts(:,ii) = P1flt2;
end

% Plot sample of faulty signal in frequency domain
f2 = Fs*(0:(samplelenght/2))/samplelenght;
subplot(2,4,3);
plot(f2, faultyts(:,testsample), 'b');
ylim([0,ylimmax]);
xlim([1,maxHz]);
title('Faulty motor frequency domain sample');
xlabel('Hz')

% Plot sample of normal signal in frequency domain
subplot(2,4,7);
plot(f2, normalts(:,testsample), 'r');
ylim([0,ylimmax]);
xlim([1,maxHz]);
title('Normal motor frequency domain sample');
xlabel('Hz')

% Reduce faulty signal sample to number of features
maxindx = find(f2 > maxHz,1)-1;

redsampleflt = faultyts(1:maxindx,:); % Reduced set of samples.
redsamplenrml = normalts(1:maxindx,:); % Reduced set of samples.
outfaulty = zeros(features, samples);
outnormal = zeros(features, samples);
var01 = floor(size(redsampleflt,1)/features);
for ii = 1:samples
    for jj = 1:features
        outfaulty(jj,ii) = max(redsampleflt((((jj-1)*var01)+1):(jj*var01),ii));
        outnormal(jj,ii) = max(redsamplenrml((((jj-1)*var01)+1):(jj*var01),ii));
    end    
end

% Plot downsampled faulty motor frequency
y = linspace(0, maxHz, features);
subplot(2,4,4);
h = stem(y, outfaulty(:,testsample), 'b');
set(h, 'Marker', 'none')
ylim([0,ylimmax]);
xlim([1,maxHz]);
title('Faulty motor frequency domain downsampled');
xlabel('Hz')

% Plot downsampled normal motor frequency
subplot(2,4,8);
h = stem(y, outnormal(:,testsample), 'r');
set(h, 'Marker', 'none')
ylim([0,ylimmax]);
xlim([1,maxHz]);
title('Normal motor frequency domain downsampled');
xlabel('Hz')

FLT = fft(fltvector);   %FFT transformation for faulty signal.
NRML = fft(nrmlvector); %FFT transformation for normal signal.

% Compute the two-sided spectrum P2. Then compute the single-sided spectrum
% P1 based on P2 and the even-valued signal length L.

% For faulty signal
P2flt = abs(FLT/Lsgn);
P1flt = P2flt(1:Lsgn/2+1);
P1flt(2:end-1) = 2*P1flt(2:end-1);

% For normal signal
P2nrml = abs(NRML/Lsgn);
P1nrml = P2nrml(1:Lsgn/2+1);
P1nrml(2:end-1) = 2*P1nrml(2:end-1);

%% PLOT REST OF THE STUFF

% Define the frequency domain f and plot the single-sided amplitude
% spectrum P1flt.
f = Fs*(0:(Lsgn/2))/Lsgn;
subplot(2,4,2);
plot(f, P1flt, 'b');
ylim([0,ylimmax]);
xlim([1,maxHz]);
title('Faulty motor frequency domain');
xlabel('Hz')

subplot(2,4,6);
plot(f, P1nrml, 'r');
ylim([0,ylimmax]);
xlim([1,maxHz]);
title('Normal motor frequency domain');
xlabel('Hz')
%% PREPARING DATA FOR NEURAL NETWORK ALGORITHMS

motorInputs = [outfaulty outnormal]; % Input data for ML algorithm. Faulty signals are first, then comes normal signals.
motorTargets = zeros(2,size(outfaulty,2)+size(outnormal,2)); % Labels for signals. Two classes.
motorTargets(2,1:size(outfaulty,2)) = 1;
motorTargets(1,size(outfaulty,2)+1:end) = 1;
%% NEURAL NETWORK

% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created 27-Jun-2021 19:17:15
%
% This script assumes these variables are defined:
%
%   motorInputs - input data.
%   motorTargets - target data.

x = motorInputs;
t = motorTargets;

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainParam.epochs = 2; % Reduce number of epochs to 2. Otherwise network will perform too well.

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)
% figure, plotconfusion(t,y)
% figure, plotroc(t,y)
