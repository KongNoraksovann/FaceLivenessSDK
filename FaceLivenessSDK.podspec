Pod::Spec.new do |liveness|
  liveness.name             = 'FaceLivenessSDK'
  liveness.version          = '1.0.0'
  liveness.license          = { :type => 'BSD' }
  liveness.homepage         = 'https://github.com/KongNoraksovann/FaceLivenessSDK'
  liveness.authors          = { 'Kong Noraksovann' => 'kongnoraksovann247@gmail.com' }
  liveness.summary          = 'Acleda FaceLivenessSDK for iOS.'
  liveness.source = { :git => 'git@github.com:KongNoraksovann/FaceLivenessSDK.git', :tag => 'v1.0.0' }
  liveness.module_name      = 'FaceLivenessSDK'
  liveness.swift_version    = '5.0'
  liveness.static_framework = true

  # ✅ iOS only
  liveness.ios.deployment_target = '12.0'

  # ✅ Source files
  liveness.source_files     = 'FaceLivenessSDK/**/*.{h,m,swift}'

  # ✅ Resources
  liveness.resources        = 'FaceLivenessSDK/**/*.{onnx,bundle,xcassets}'

  # ✅ Dependencies
  liveness.dependency 'onnxruntime-objc', '~> 1.18.0'
  liveness.dependency 'GoogleMLKit/Vision'
  liveness.dependency 'GoogleMLKit/FaceDetection'

  # ✅ Optional vendored framework
  # If you're using a prebuilt .framework (like built with Xcode), include it here.
  # Remove this if you're only using source files.
  # liveness.vendored_frameworks = 'FaceLivenessSDK.framework'
end
