import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage.color import rgb2gray
from scipy.signal import find_peaks
from functools import lru_cache

class ImageSearchUtils:
    def __init__(self):
        # HSV color ranges for better detection
        self.color_ranges = {
            'red': {
                'ranges': [
                    [(0, 70, 50), (10, 255, 255)],     # Red range 1
                    [(170, 70, 50), (180, 255, 255)]   # Red range 2
                ]
            },
            'blue': {
                'ranges': [[(100, 50, 50), (130, 255, 255)]]
            },
            'green': {
                'ranges': [[(35, 50, 50), (85, 255, 255)]]
            },
            'yellow': {
                'ranges': [[(20, 50, 50), (35, 255, 255)]]
            },
            'purple': {
                'ranges': [[(130, 50, 50), (155, 255, 255)]]
            },
            'orange': {
                'ranges': [[(10, 50, 50), (20, 255, 255)]]
            },
            'pink': {
                'ranges': [[(155, 50, 50), (170, 255, 255)]]
            },
            'brown': {
                'ranges': [[(0, 40, 10), (20, 100, 100)]]
            }
        }
        self.session = requests.Session()
        self.image_cache = {}

    def get_color_weights(self, image_path):
        try:
            # Check if the image_path is a URL
            if image_path.startswith('http://') or image_path.startswith('https://'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)

            # Convert to numpy array
            img_array = np.array(img)
            
            # Convert to HSV
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Cluster colors in HSV space
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(hsv.reshape(-1, 3))
            
            # Calculate weights
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            color_weights = {}
            total_pixels = len(labels)
            
            for color, count in zip(colors, counts):
                color_name = self._get_closest_color(color)
                if color_name:
                    weight = count / total_pixels
                    if weight > 0.1:  # Minimum 10% threshold
                        if color_name in color_weights:
                            color_weights[color_name] += weight
                        else:
                            color_weights[color_name] = weight
            
            # Normalize weights
            if color_weights:
                total = sum(color_weights.values())
                color_weights = {k: round(v/total, 2) for k, v in color_weights.items()}
            
            print(f"Debug - Detected color weights: {color_weights}")
            return color_weights
            
        except Exception as e:
            print(f"Color weight extraction error: {e}")
            return None

    def _get_closest_color(self, hsv):
        """Find closest matching color in HSV space"""
        h, s, v = hsv
        
        # Normalize hue to 0-180 range
        h = int(h % 180)
        s = int(s)
        v = int(v)
        
        for color_name, color_info in self.color_ranges.items():
            for (h_min, s_min, v_min), (h_max, s_max, v_max) in color_info['ranges']:
                if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                    return color_name
        
        return None

    def calculate_color_similarity(self, weights1, weights2):
        """Calculate similarity between two color weight distributions"""
        if not weights1 or not weights2:
            return 0.0
        
        similarity = 0.0
        all_colors = set(weights1.keys()) | set(weights2.keys())
        
        for color in all_colors:
            w1 = weights1.get(color, 0.0)
            w2 = weights2.get(color, 0.0)
            similarity += 1.0 - abs(w1 - w2)
        
        similarity = similarity / len(all_colors)
        print(f"Debug - Color similarity score: {similarity}")
        return similarity

    def calculate_pattern_similarity(self, patterns1, patterns2):
        """Calculate similarity between two pattern distributions"""
        if not patterns1 or not patterns2:
            return 0.0
        
        similarity = 0.0
        all_patterns = set(patterns1.keys()) | set(patterns2.keys())
        
        for pattern in all_patterns:
            w1 = patterns1.get(pattern, 0.0)
            w2 = patterns2.get(pattern, 0.0)
            similarity += 1.0 - abs(w1 - w2)
        
        similarity = similarity / len(all_patterns)
        print(f"Debug - Pattern similarity score: {similarity}")
        return similarity
    
    @lru_cache(maxsize=100)
    def _load_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Image loading error: {e}")
            return None

    def _load_url_image(self, url):
        """Load and cache image from URL"""
        try:
            if url in self.image_cache:
                return self.image_cache[url]

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to numpy array
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                self.image_cache[url] = img
                return img
            return None
        except requests.exceptions.RequestException as e:
            print(f"HTTP request error: {e}")
            return None
        except Exception as e:
            print(f"Image decoding error: {e}")
            return None
    
    def calculate_combined_similarity(self, color_weights1, color_weights2, pattern_weights1, pattern_weights2):
        """Calculate combined similarity between color and pattern distributions"""
        color_similarity = self.calculate_color_similarity(color_weights1, color_weights2)
        pattern_similarity = self.calculate_pattern_similarity(pattern_weights1, pattern_weights2)
        
        combined_similarity = (color_similarity + pattern_similarity) / 2
        print(f"Debug - Combined similarity score: {combined_similarity}")
        return combined_similarity

    def detect_pattern(self, image_path):
        """Detect patterns in image from file or URL"""
        try:
            # Use same loading logic as color detection
            if image_path.startswith('http://') or image_path.startswith('https://'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)

            # Convert to numpy array
            img_array = np.array(img)
            
            # Convert to grayscale for pattern detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Pattern detection
            patterns = {
                'striped': self._detect_stripes(gray),
                'polka_dot': self._detect_dots(gray),
                'checkered': self._detect_grid(gray),
                'floral': self._detect_organic(gray),
                'geometric': self._detect_shapes(gray),
                'plain': self._detect_solid(gray)
            }
            
            # Normalize confidence scores
            total = sum(patterns.values())
            if total > 0:
                patterns = {k: round(v/total, 2) for k, v in patterns.items()}
            
            print(f"Debug - Detected patterns: {patterns}")
            return patterns

        except Exception as e:
            print(f"Pattern detection error: {e}")
            return None

# Rest of the methods for pattern detection would need to be implemented
# These placeholders ensure the code structure remains similar
def _detect_stripes(self, gray):
    return 0.5

def _detect_dots(self, gray):
    return 0.5

def _detect_grid(self, gray):
    return 0.5

def _detect_organic(self, gray):
    return 0.5

def _detect_shapes(self, gray):
    return 0.5

def _detect_solid(self, gray):
    return 0.5

# Create singleton instance
image_search_utils = ImageSearchUtils()

class PatternDetector:
    def __init__(self):
        self.pattern_thresholds = {
            'striped': 0.7,      # Strong directional lines
            'polka_dot': 0.6,    # Circular repeating patterns
            'floral': 0.5,       # Organic, curved shapes
            'checkered': 0.65,   # Grid-like patterns
            'geometric': 0.55,    # Regular shapes
            'animal': 0.45,      # Irregular natural patterns
            'abstract': 0.4,     # Random, non-geometric patterns
            'plain': 0.3         # Solid colors, minimal texture
        }
        self.session = requests.Session()

    def detect_pattern(self, image_path):
        """Detect patterns in image from file or URL"""
        try:
            # Load image based on path type
            if isinstance(image_path, str) and (image_path.startswith('http://') or image_path.startswith('https://')):
                response = self.session.get(image_path, timeout=10)
                response.raise_for_status()
                
                # Convert to numpy array using PIL first
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(image_path)

            if img is None:
                print(f"Failed to load image: {image_path}")
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Pattern detection with existing methods
            hog_features, hog_image = self._extract_hog_features(gray)
            directionality = self._analyze_directionality(hog_image)
            periodicity = self._detect_periodicity(gray)
            geometry = self._detect_geometric_patterns(gray)
            texture = self._analyze_texture_complexity(gray)
            
            pattern_probabilities = {
                'plain': max(1 - texture - directionality - periodicity, 0),
                'striped': directionality * 0.8,
                'polka_dot': periodicity * 0.7,
                'floral': self._detect_floral_pattern(img) * 0.6,
                'checkered': self._detect_checkered_pattern(gray) * 0.7,
                'geometric': geometry * 0.6,
                'animal': self._detect_animal_print(img) * 0.5,
                'abstract': texture * (1 - geometry) * 0.4
            }
            
            # Normalize probabilities
            total = sum(pattern_probabilities.values())
            if total > 0:
                pattern_probabilities = {k: round(v/total, 2) for k, v in pattern_probabilities.items()}
            
            print(f"Debug - Pattern detection for {image_path}: {pattern_probabilities}")
            return pattern_probabilities

        except Exception as e:
            print(f"Pattern detection error for {image_path}: {e}")
            return None

    def _extract_hog_features(self, gray_image):
        """Extract Histogram of Oriented Gradients (HOG) features"""
        hog_features, hog_image = hog(
            gray_image, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True
        )
        return hog_features, hog_image

    def _analyze_directionality(self, hog_image):
        """Analyze texture directionality"""
        # Use Fourier Transform to analyze directional patterns
        f_transform = np.fft.fft2(hog_image)
        f_shifted = np.fft.fftshift(f_transform)
        
        # Compute magnitude spectrum
        magnitude_spectrum = 20*np.log(np.abs(f_shifted))
        
        # Check for strong linear/directional components
        directionality_score = np.max(magnitude_spectrum) / np.mean(magnitude_spectrum)
        return min(directionality_score / 10, 1)  # Normalize

    def _detect_periodicity(self, gray_image):
        """Detect periodic structures like polka dots"""
        # Compute autocorrelation
        autocorr = cv2.matchTemplate(gray_image, gray_image, cv2.TM_CCORR_NORMED)
        
        # Find peaks in horizontal and vertical directions
        h_peaks, _ = find_peaks(autocorr.mean(axis=0), distance=20)
        v_peaks, _ = find_peaks(autocorr.mean(axis=1), distance=20)
        
        # Periodicity score based on number and regularity of peaks
        periodicity_score = (len(h_peaks) + len(v_peaks)) / (gray_image.shape[1] + gray_image.shape[0])
        return min(periodicity_score, 1)

    def _detect_floral_pattern(self, image):
        """Detect floral/organic patterns"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect color variance and segmentation
        h, s, v = cv2.split(hsv)
        color_variance = np.std(h)
        
        # Use color distribution to hint at floral patterns
        floral_score = color_variance / 180  # Normalize hue variance
        return min(floral_score, 1)

    def _detect_checkered_pattern(self, gray_image):
        # Detect grid-like structures using edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            horizontal = 0
            vertical = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1))
                if angle < np.pi/4 or angle > 3*np.pi/4:
                    horizontal += 1
                else:
                    vertical += 1
            
            return min((horizontal * vertical) / (len(lines) ** 2), 1)
        return 0

    def _detect_geometric_patterns(self, gray_image):
        # Detect regular shapes using contour analysis
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
            
        geometric_score = 0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) >= 3 and len(approx) <= 8:
                geometric_score += 1
                
        return min(geometric_score / len(contours), 1)

    def _detect_animal_print(self, image):
        # Convert to grayscale and analyze texture variation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Calculate local variance
        local_var = cv2.Laplacian(blur, cv2.CV_64F).var()
        
        # Detect spots/stripes
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
            
        irregularity = sum(cv2.contourArea(cnt) for cnt in contours) / (gray.shape[0] * gray.shape[1])
        return min((local_var * irregularity) / 1000, 1)

    def _analyze_texture_complexity(self, gray_image):
        # Analyze texture using GLCM (Gray-Level Co-occurrence Matrix)
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return min(np.mean(gradient_magnitude) / 128, 1)

# Singleton instance
pattern_detector = PatternDetector()
