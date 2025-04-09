import Foundation
import os.log

/**
 * Utility class for consistent logging in the FaceLivenessSDK.
 * Supports debug, info, warning, and error levels with configurable debug output.
 */
@objc public class LogUtils: NSObject {
    private static let SDK_TAG_PREFIX = "FaceSDK-"
    private static var isDebugEnabled = false
    
    @objc public static func setDebugEnabled(_ enabled: Bool) {
        isDebugEnabled = enabled
    }
    
    @objc public static func d(_ tag: String, _ message: String) {
        if isDebugEnabled {
            if #available(iOS 14.0, *) {
                let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
                logger.debug("\(message)")
            } else {
                NSLog("[\(SDK_TAG_PREFIX)\(tag)] [DEBUG] \(message)")
            }
        }
    }
    
    @objc public static func i(_ tag: String, _ message: String) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.info("\(message)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [INFO] \(message)")
        }
    }
    
    @objc public static func w(_ tag: String, _ message: String) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.warning("\(message)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [WARNING] \(message)")
        }
    }
    
    @objc public static func e(_ tag: String, _ message: String) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.error("\(message)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [ERROR] \(message)")
        }
    }
    
    @objc public static func e(_ tag: String, _ message: String, _ error: Error) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.error("\(message): \(error.localizedDescription)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [ERROR] \(message): \(error.localizedDescription)")
        }
    }
}
