// Function to set default playback speed for all videos
function setDefaultPlaybackSpeed() {
    // Get all video elements
    const videos = document.querySelectorAll('video');
    
    // Set playback rate for each video
    videos.forEach(video => {
      // Set initial playback rate to 2x
      video.defaultPlaybackRate = 2.0;
      video.playbackRate = 2.0;
      
      // Add loadedmetadata event listener to ensure playback rate is set
      video.addEventListener('loadedmetadata', function() {
        this.playbackRate = 2.0;
      });
      
      // Add play event listener to maintain 2x speed when video starts playing
      video.addEventListener('play', function() {
        if (this.playbackRate !== 2.0) {
          this.playbackRate = 2.0;
        }
      });
    });
  }
  
  // Run when DOM content is loaded
  document.addEventListener('DOMContentLoaded', setDefaultPlaybackSpeed);