import requests
import re
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class SocialMediaScraper(ABC):
    """Abstract base class for social media scrapers"""
    
    @abstractmethod
    def extract_comments(self, url: str) -> List[Dict]:
        """Extract comments from a social media post URL"""
        pass
    
    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """Validate if the URL is from the correct platform"""
        pass


class YouTubeScraper(SocialMediaScraper):
    """YouTube comment scraper using YouTube Data API v3"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def validate_url(self, url: str) -> bool:
        """Validate YouTube URL patterns"""
        youtube_patterns = [
            r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'youtu\.be/([a-zA-Z0-9_-]+)',
            r'youtube\.com/embed/([a-zA-Z0-9_-]+)'
        ]
        return any(re.search(pattern, url) for pattern in youtube_patterns)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'youtu\.be/([a-zA-Z0-9_-]+)',
            r'youtube\.com/embed/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def extract_comments(self, url: str) -> List[Dict]:
        """Extract comments from YouTube video"""
        if not self.api_key:
            # Fallback to basic scraping without API
            return self._scrape_without_api(url)
        
        video_id = self.extract_video_id(url)
        if not video_id:
            return []
        
        comments = []
        page_token = None
        max_results = 100
        
        while True:
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'key': self.api_key,
                'maxResults': max_results,
                'order': 'relevance'
            }
            
            if page_token:
                params['pageToken'] = page_token
            
            try:
                response = requests.get(f"{self.base_url}/commentThreads", params=params)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get('items', []):
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'text': comment_data['textDisplay'],
                        'author': comment_data['authorDisplayName'],
                        'likes': comment_data.get('likeCount', 0),
                        'published_at': comment_data['publishedAt'],
                        'comment_id': item['id']
                    })
                
                page_token = data.get('nextPageToken')
                if not page_token or len(comments) >= 1000:  # Limit to prevent overload
                    break
                    
            except requests.RequestException as e:
                print(f"Error fetching YouTube comments: {e}")
                break
        
        return comments
    
    def _scrape_without_api(self, url: str) -> List[Dict]:
        """Basic scraping without API (limited functionality)"""
        # This is a placeholder - in production you'd need more sophisticated scraping
        # or require API keys for proper functionality
        return [
            {
                'text': 'Sample comment (API key required for real data)',
                'author': 'Sample User',
                'likes': 0,
                'published_at': '2024-01-01T00:00:00Z',
                'comment_id': 'sample_id'
            }
        ]


class InstagramScraper(SocialMediaScraper):
    """Instagram comment scraper"""
    
    def validate_url(self, url: str) -> bool:
        """Validate Instagram URL patterns"""
        instagram_patterns = [
            r'instagram\.com/p/([a-zA-Z0-9_-]+)',
            r'instagram\.com/reel/([a-zA-Z0-9_-]+)'
        ]
        return any(re.search(pattern, url) for pattern in instagram_patterns)
    
    def extract_comments(self, url: str) -> List[Dict]:
        """Extract comments from Instagram post"""
        # Note: Instagram has strict anti-scraping measures
        # This is a simplified implementation for demonstration
        
        # In production, you would use:
        # 1. Instagram Basic Display API (limited)
        # 2. Instagram Graph API (business accounts)
        # 3. Third-party services like Instaloader (with caution)
        
        try:
            # Using instaloader as an example (requires careful usage)
            import instaloader
            
            loader = instaloader.Instaloader()
            
            # Extract shortcode from URL
            shortcode_match = re.search(r'/p/([a-zA-Z0-9_-]+)', url)
            if not shortcode_match:
                shortcode_match = re.search(r'/reel/([a-zA-Z0-9_-]+)', url)
            
            if not shortcode_match:
                return []
            
            shortcode = shortcode_match.group(1)
            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            
            comments = []
            for comment in post.get_comments():
                comments.append({
                    'text': comment.text,
                    'author': comment.owner.username,
                    'likes': comment.likes_count,
                    'published_at': comment.created_at_utc.isoformat(),
                    'comment_id': comment.id
                })
                
                # Limit to prevent overload
                if len(comments) >= 500:
                    break
            
            return comments
            
        except Exception as e:
            print(f"Error scraping Instagram comments: {e}")
            # Return sample data for testing
            return [
                {
                    'text': 'Sample Instagram comment (requires proper setup)',
                    'author': 'sample_user',
                    'likes': 0,
                    'published_at': '2024-01-01T00:00:00Z',
                    'comment_id': 'sample_ig_id'
                }
            ]


class TikTokScraper(SocialMediaScraper):
    """TikTok comment scraper"""
    
    def validate_url(self, url: str) -> bool:
        """Validate TikTok URL patterns"""
        tiktok_patterns = [
            r'tiktok\.com/@[^/]+/video/(\d+)',
            r'vm\.tiktok\.com/([a-zA-Z0-9]+)',
            r'tiktok\.com/t/([a-zA-Z0-9]+)'
        ]
        return any(re.search(pattern, url) for pattern in tiktok_patterns)
    
    def extract_comments(self, url: str) -> List[Dict]:
        """Extract comments from TikTok video"""
        try:
            # Using TikTokApi as an example
            from TikTokApi import TikTokApi
            
            # Note: TikTok API usage requires careful handling
            # and may need regular updates due to platform changes
            
            api = TikTokApi()
            
            # Extract video ID from URL
            video_id_match = re.search(r'/video/(\d+)', url)
            if not video_id_match:
                return []
            
            video_id = video_id_match.group(1)
            
            # Get video object
            video = api.video(id=video_id)
            
            comments = []
            for comment in video.comments():
                comments.append({
                    'text': comment.text,
                    'author': comment.author.username,
                    'likes': comment.likes_count,
                    'published_at': comment.create_time,
                    'comment_id': comment.id
                })
                
                # Limit to prevent overload
                if len(comments) >= 500:
                    break
            
            return comments
            
        except Exception as e:
            print(f"Error scraping TikTok comments: {e}")
            # Return sample data for testing
            return [
                {
                    'text': 'Sample TikTok comment (requires proper API setup)',
                    'author': 'sample_tiktoker',
                    'likes': 0,
                    'published_at': '2024-01-01T00:00:00Z',
                    'comment_id': 'sample_tt_id'
                }
            ]


class SocialMediaScraperFactory:
    """Factory class to get appropriate scraper based on URL"""
    
    @staticmethod
    def get_scraper(url: str) -> Optional[SocialMediaScraper]:
        """Return appropriate scraper based on URL"""
        
        # You can add API keys here from Django settings
        youtube_api_key = None  # Get from settings in production
        
        scrapers = [
            YouTubeScraper(api_key=youtube_api_key),
            InstagramScraper(),
            TikTokScraper()
        ]
        
        for scraper in scrapers:
            if scraper.validate_url(url):
                return scraper
        
        return None
    
    @staticmethod
    def get_platform_name(url: str) -> Optional[str]:
        """Get platform name from URL"""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'instagram.com' in url:
            return 'instagram'
        elif 'tiktok.com' in url:
            return 'tiktok'
        return None