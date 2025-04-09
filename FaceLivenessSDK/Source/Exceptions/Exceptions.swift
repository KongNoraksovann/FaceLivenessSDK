import Foundation

@objc public class FaceLivenessException: NSError {
    @objc public convenience init(_ message: String, _ cause: Error? = nil) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message,
            NSUnderlyingErrorKey: cause as Any
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1000, userInfo: userInfo)
    }
}

@objc public class ModelLoadingException: FaceLivenessException {
    @objc public convenience init(_ message: String, _ cause: Error? = nil) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message,
            NSUnderlyingErrorKey: cause as Any
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1001, userInfo: userInfo)
    }
}

@objc public class InvalidImageException: FaceLivenessException {
    @objc public convenience init(_ message: String) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1002, userInfo: userInfo)
    }
}

@objc public class FaceDetectionException: FaceLivenessException {
    @objc public convenience init(_ message: String, _ cause: Error? = nil) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message,
            NSUnderlyingErrorKey: cause as Any
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1003, userInfo: userInfo)
    }
}

@objc public class LivenessException: FaceLivenessException {
    @objc public convenience init(_ message: String, _ cause: Error? = nil) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message,
            NSUnderlyingErrorKey: cause as Any
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1004, userInfo: userInfo)
    }
}

@objc public class OcclusionDetectionException: FaceLivenessException {
    @objc public convenience init(_ message: String, _ cause: Error? = nil) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message,
            NSUnderlyingErrorKey: cause as Any
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1005, userInfo: userInfo)
    }
}

@objc public class QualityCheckException: FaceLivenessException {
    @objc public convenience init(_ message: String, _ cause: Error? = nil) {
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message,
            NSUnderlyingErrorKey: cause as Any
        ]
        self.init(domain: "com.acleda.facelivenesssdk", code: 1006, userInfo: userInfo)
    }
}
