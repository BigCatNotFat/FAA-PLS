class Logger:
    # 单例实例
    _instance = None
    
    # 日志级别定义
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    
    # 日志级别映射
    LEVEL_MAP = {
        "DEBUG": DEBUG,
        "INFO": INFO,
        "WARNING": WARNING,
        "ERROR": ERROR
    }
    
    # ANSI颜色代码
    COLORS = {
        "调试": "\033[36m",  # 青色
        "信息": "\033[32m",  # 绿色
        "警告": "\033[33m",  # 黄色
        "错误": "\033[31m",  # 红色
        "重置": "\033[0m",   # 重置颜色
        "内容": "\033[37m"   # 白色用于内容
    }
    
    # 文件名颜色映射
    FILE_COLORS = {
        "faa_pls.py": "\033[35m",      # 紫色
        "opt.py": "\033[34m",          # 蓝色
        "config.py": "\033[33m",       # 黄色
        "faa_rot_model.py": "\033[36m",# 青色
        "logger.py": "\033[90m",       # 灰色
        "default": "\033[37m"          # 默认白色
    }
    
    @staticmethod
    def get_instance(config=None):
        """获取日志单例实例"""
        if Logger._instance is None:
            Logger._instance = Logger(config)
        elif config is not None:
            Logger._instance.update_config(config)
        return Logger._instance
    
    def __init__(self, config=None):
        """初始化日志类"""
        self.enabled = True
        self.level = Logger.INFO
        self.log_to_file = False
        self.log_file_path = "logs/app.log"
        self.use_colors = True  # 默认启用颜色
        self.console_output = True  # 默认启用控制台输出
        
        if config is not None:
            self.update_config(config)
    
    def update_config(self, config):
        """根据配置更新日志设置"""
        if hasattr(config, 'log_enabled'):
            self.enabled = config.log_enabled
        
        if hasattr(config, 'log_level'):
            self.level = self.LEVEL_MAP.get(config.log_level, Logger.INFO)
            
        if hasattr(config, 'log_to_file'):
            self.log_to_file = config.log_to_file
            
        if hasattr(config, 'log_file_path'):
            self.log_file_path = config.log_file_path
            
        if hasattr(config, 'log_use_colors'):
            self.use_colors = config.log_use_colors
            
        if hasattr(config, 'log_console_output'):
            self.console_output = config.log_console_output
    
    def _write_log(self, level_str, *args):
        """写入日志，支持多参数"""
        from datetime import datetime
        import inspect
        import os
        
        # 获取调用栈信息，向上查找直到找到非logger.py的调用者
        frame = inspect.currentframe().f_back.f_back  # 跳过_write_log和debug/info等函数
        
        # 继续向上查找调用栈，直到找到非logger.py的调用者
        while frame and os.path.basename(frame.f_code.co_filename) == 'logger.py':
            frame = frame.f_back
            
        # 如果找不到非logger.py的调用者，就使用当前帧
        if not frame:
            frame = inspect.currentframe().f_back.f_back
            
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建日志头部，包含位置信息
        if self.use_colors:
            level_color = self.COLORS.get(level_str, "")
            reset_color = self.COLORS["重置"]
            content_color = self.COLORS["内容"]
            # 获取文件对应的颜色
            file_color = self.FILE_COLORS.get(filename, self.FILE_COLORS["default"])
            log_header = f"{level_color}[{current_time}] [{level_str}] {file_color}[{filename}:{lineno}]{level_color}"
        else:
            log_header = f"[{current_time}] [{level_str}] [{filename}:{lineno}]"
        
        # 将所有参数转换为字符串并连接
        log_content = "\n".join(str(arg) for arg in args)
        
        # 组合完整日志消息：头部信息在第一行，内容从第二行开始
        if self.use_colors:
            log_message = f"{log_header}\n{content_color}{log_content}{reset_color}"
            console_message = log_message  # 控制台显示带颜色
            file_message = f"[{current_time}] [{level_str}] [{filename}:{lineno}]\n{log_content}"  # 文件中不含颜色代码
        else:
            log_message = f"{log_header}\n{log_content}"
            console_message = log_message
            file_message = log_message
        
        # 输出到控制台（如果启用）
        if self.console_output:
            print(console_message)
        
        # 如果配置了文件输出，也写入文件（即使控制台输出被禁用）
        if self.log_to_file:
            try:
                # 确保日志目录存在
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(file_message + "\n")
            except Exception as e:
                error_msg = f"[{current_time}] [错误] [{filename}:{lineno}] 无法写入日志文件: {e}"
                if self.use_colors:
                    error_msg = f"{self.COLORS['错误']}{error_msg}{self.COLORS['重置']}"
                print(error_msg)  # 即使控制台输出被禁用，也打印文件写入错误
    
    def debug(self, *args):
        """调试级别日志"""
        if self.enabled and self.level <= Logger.DEBUG:
            self._write_log("调试", *args)
    
    def info(self, *args):
        """信息级别日志"""
        if self.enabled and self.level <= Logger.INFO:
            self._write_log("信息", *args)
    
    def warning(self, *args):
        """警告级别日志"""
        if self.enabled and self.level <= Logger.WARNING:
            self._write_log("警告", *args)
    
    def error(self, *args):
        """错误级别日志"""
        if self.enabled and self.level <= Logger.ERROR:
            self._write_log("错误", *args)

# 创建默认实例
_logger = Logger()

# 模块级别的函数，不需要获取实例就可以使用
def init_logger(config):
    """使用配置初始化日志系统"""
    global _logger
    _logger = Logger.get_instance(config)

def debug(*args):
    """模块级别的调试日志函数"""
    _logger.debug(*args)

def info(*args):
    """模块级别的信息日志函数"""
    _logger.info(*args)

def warning(*args):
    """模块级别的警告日志函数"""
    _logger.warning(*args)

def error(*args):
    """模块级别的错误日志函数"""
    _logger.error(*args)
