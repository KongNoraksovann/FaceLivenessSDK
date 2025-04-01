#!/bin/bash

# Exit when any command fails
set -e

# Parameters
FRAMEWORK_NAME="FaceLivenessSDK"
SCHEME="FaceLivenessSDK"
OUTPUT_DIR="${PWD}/xcframework_output"
DERIVED_DATA_PATH="${PWD}/DerivedData"

# Create necessary directories
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
rm -rf "${DERIVED_DATA_PATH}"
mkdir -p "${DERIVED_DATA_PATH}"

# Install dependencies with CocoaPods
echo "Installing CocoaPods dependencies..."
pod install

# Clean the project first
echo "Cleaning project..."
xcodebuild clean \
    -workspace "${FRAMEWORK_NAME}.xcworkspace" \
    -scheme "${SCHEME}" \
    -configuration Release

# Build for iOS devices
echo "Building ${FRAMEWORK_NAME} for iOS devices..."
xcodebuild archive \
    -workspace "${FRAMEWORK_NAME}.xcworkspace" \
    -scheme "${SCHEME}" \
    -configuration Release \
    -sdk iphoneos \
    -destination "generic/platform=iOS" \
    -archivePath "${OUTPUT_DIR}/${FRAMEWORK_NAME}_iOS.xcarchive" \
    -derivedDataPath "${DERIVED_DATA_PATH}" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    SWIFT_EMIT_MODULE_INTERFACE=YES \
    ONLY_ACTIVE_ARCH=NO \
    ENABLE_USER_SCRIPT_SANDBOXING=NO

# Build for iOS simulators
echo "Building ${FRAMEWORK_NAME} for iOS simulators..."
xcodebuild archive \
    -workspace "${FRAMEWORK_NAME}.xcworkspace" \
    -scheme "${SCHEME}" \
    -configuration Release \
    -sdk iphonesimulator \
    -destination "generic/platform=iOS Simulator" \
    -archivePath "${OUTPUT_DIR}/${FRAMEWORK_NAME}_iOSSimulator.xcarchive" \
    -derivedDataPath "${DERIVED_DATA_PATH}" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    SWIFT_EMIT_MODULE_INTERFACE=YES \
    ONLY_ACTIVE_ARCH=NO \
    ENABLE_USER_SCRIPT_SANDBOXING=NO

# Create XCFramework
echo "Creating XCFramework..."
xcodebuild -create-xcframework \
    -framework "${OUTPUT_DIR}/${FRAMEWORK_NAME}_iOS.xcarchive/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
    -framework "${OUTPUT_DIR}/${FRAMEWORK_NAME}_iOSSimulator.xcarchive/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
    -output "${OUTPUT_DIR}/${FRAMEWORK_NAME}.xcframework"

echo "Done! XCFramework is available at: ${OUTPUT_DIR}/${FRAMEWORK_NAME}.xcframework"
