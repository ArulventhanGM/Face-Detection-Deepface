/**
 * Futuristic Background Effects
 * Implements animated line patterns and glowing particle effects
 */

class FuturisticBackgroundEffects {
    constructor() {
        this.particles = [];
        this.particleCanvas = null;
        this.particleContext = null;
        this.animationFrameId = null;
        this.mouseX = 0;
        this.mouseY = 0;
        
        this.init();
    }
    
    init() {
        // Create particle canvas
        this.createParticleCanvas();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Create initial particles
        this.createInitialParticles();
        
        // Start animation
        this.animate();
    }
    
    createParticleCanvas() {
        this.particleCanvas = document.createElement('canvas');
        this.particleCanvas.classList.add('particle-canvas');
        this.particleContext = this.particleCanvas.getContext('2d');
        
        // Style the canvas
        this.particleCanvas.style.position = 'fixed';
        this.particleCanvas.style.top = '0';
        this.particleCanvas.style.left = '0';
        this.particleCanvas.style.width = '100%';
        this.particleCanvas.style.height = '100%';
        this.particleCanvas.style.pointerEvents = 'none';
        this.particleCanvas.style.zIndex = '-1';
        
        // Add to DOM
        document.body.appendChild(this.particleCanvas);
        
        // Set canvas dimensions
        this.resizeCanvas();
    }
    
    setupEventListeners() {
        // Resize event
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
        
        // Mouse move event
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            
            // Create particle at mouse position occasionally
            if (Math.random() > 0.92) {
                this.createParticle(
                    e.clientX,
                    e.clientY,
                    Math.random() * 2 + 1,
                    this.getRandomColor()
                );
            }
        });
    }
    
    resizeCanvas() {
        this.particleCanvas.width = window.innerWidth;
        this.particleCanvas.height = window.innerHeight;
    }
    
    createInitialParticles() {
        const particleCount = Math.floor(window.innerWidth * window.innerHeight / 10000);
        
        for (let i = 0; i < particleCount; i++) {
            this.createParticle(
                Math.random() * window.innerWidth,
                Math.random() * window.innerHeight,
                Math.random() * 2 + 0.5,
                this.getRandomColor()
            );
        }
    }
    
    createParticle(x, y, size, color) {
        const particle = {
            x: x,
            y: y,
            size: size,
            color: color,
            speedX: (Math.random() - 0.5) * 0.5,
            speedY: (Math.random() - 0.5) * 0.5,
            life: Math.random() * 100 + 100,
            maxLife: Math.random() * 100 + 100
        };
        
        this.particles.push(particle);
        
        // Cap particles count
        if (this.particles.length > 300) {
            this.particles.shift();
        }
    }
    
    getRandomColor() {
        const colors = [
            'rgba(230, 28, 81, 0.6)',   // Crimson
            'rgba(99, 50, 230, 0.6)',    // Violet
            'rgba(0, 185, 199, 0.6)'     // Teal
        ];
        
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        
        // Clear canvas
        this.particleContext.clearRect(0, 0, this.particleCanvas.width, this.particleCanvas.height);
        
        // Update and draw particles
        this.updateParticles();
        
        // Draw grid effect
        this.drawGridEffect();
    }
    
    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            
            // Update position
            p.x += p.speedX;
            p.y += p.speedY;
            
            // Update life
            p.life--;
            
            // Fade out as life decreases
            const opacity = p.life / p.maxLife * 0.6;
            const rgb = p.color.slice(0, p.color.lastIndexOf(',') + 1);
            p.currentColor = `${rgb} ${opacity})`;
            
            // Remove dead particles
            if (p.life <= 0) {
                this.particles.splice(i, 1);
                continue;
            }
            
            // Draw particle
            this.particleContext.beginPath();
            this.particleContext.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.particleContext.fillStyle = p.currentColor;
            this.particleContext.fill();
            
            // Draw connections between nearby particles
            this.drawConnections(p, i);
        }
    }
    
    drawConnections(p, index) {
        for (let j = index - 1; j >= 0; j--) {
            const p2 = this.particles[j];
            const dx = p.x - p2.x;
            const dy = p.y - p2.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            // Draw line if particles are close enough
            if (dist < 100) {
                const opacity = (1 - dist / 100) * 0.2;
                this.particleContext.beginPath();
                this.particleContext.moveTo(p.x, p.y);
                this.particleContext.lineTo(p2.x, p2.y);
                this.particleContext.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
                this.particleContext.stroke();
            }
        }
    }
    
    drawGridEffect() {
        const time = Date.now() * 0.0005;
        const gridSize = 40;
        
        this.particleContext.lineWidth = 0.2;
        this.particleContext.strokeStyle = 'rgba(100, 100, 255, 0.1)';
        
        // Draw horizontal lines
        for (let y = 0; y < this.particleCanvas.height; y += gridSize) {
            const waveOffset = Math.sin(time + y * 0.01) * 5;
            
            this.particleContext.beginPath();
            this.particleContext.moveTo(0, y + waveOffset);
            this.particleContext.lineTo(this.particleCanvas.width, y + waveOffset);
            this.particleContext.stroke();
        }
        
        // Draw vertical lines
        for (let x = 0; x < this.particleCanvas.width; x += gridSize) {
            const waveOffset = Math.sin(time + x * 0.01) * 5;
            
            this.particleContext.beginPath();
            this.particleContext.moveTo(x + waveOffset, 0);
            this.particleContext.lineTo(x + waveOffset, this.particleCanvas.height);
            this.particleContext.stroke();
        }
    }
    
    // Method to add glowing sphere
    addGlowingSphere(x, y) {
        const sphere = {
            x: x || Math.random() * window.innerWidth,
            y: y || Math.random() * window.innerHeight,
            size: Math.random() * 30 + 20,
            color: this.getRandomColor(),
            speedX: (Math.random() - 0.5) * 0.3,
            speedY: (Math.random() - 0.5) * 0.3,
            life: 300,
            maxLife: 300
        };
        
        this.particles.push(sphere);
    }
    
    // Add multiple spheres
    addGlowingSpheres(count) {
        for (let i = 0; i < count; i++) {
            this.addGlowingSphere();
        }
    }
    
    // Clean up resources
    destroy() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        
        if (this.particleCanvas && this.particleCanvas.parentNode) {
            this.particleCanvas.parentNode.removeChild(this.particleCanvas);
        }
    }
}

// Initialize the background effects when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.futuristicBackgroundEffects = new FuturisticBackgroundEffects();
    
    // Add some glowing spheres
    window.futuristicBackgroundEffects.addGlowingSpheres(5);
});