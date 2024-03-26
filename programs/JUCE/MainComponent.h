#pragma once

#include "JuceHeader.h"

//==============================================================================
/*
 This component lives inside our window, and this is where you should put all
 your controls and content.
 */
class MainComponent final : public juce::OpenGLAppComponent
{
public:
  //==============================================================================
  MainComponent();
  ~MainComponent() override;
  
  //==============================================================================
  void initialise() override;
  void shutdown() override;
  void render() override;
  
  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override;
  
  void createShaders();

private:
  //==============================================================================
  struct Vertex
  {
    float position[3];
    float texCoord[2];
  };
  
  //==============================================================================
  // This class just manages the attributes that the shaders use.
  struct Attributes
  {
    explicit Attributes (OpenGLShaderProgram& shaderProgram)
    {
      position      .reset (createAttribute (shaderProgram, "position"));
      textureCoordIn.reset (createAttribute (shaderProgram, "textureCoordIn"));
    }
    
    void enable()
    {
      using namespace ::juce::gl;
      
      if (position.get() != nullptr)
      {
        glVertexAttribPointer (position->attributeID, 3, GL_FLOAT, GL_FALSE, sizeof (Vertex), nullptr);
        glEnableVertexAttribArray (position->attributeID);
      }
      
      if (textureCoordIn.get() != nullptr)
      {
        glVertexAttribPointer (textureCoordIn->attributeID, 2, GL_FLOAT, GL_FALSE, sizeof (Vertex), (GLvoid*) (sizeof (float) * 3));
        glEnableVertexAttribArray (textureCoordIn->attributeID);
      }
    }
    
    void disable()
    {
      using namespace ::juce::gl;
      
      if (position != nullptr)       glDisableVertexAttribArray (position->attributeID);
      if (textureCoordIn != nullptr) glDisableVertexAttribArray (textureCoordIn->attributeID);
    }
    
    std::unique_ptr<OpenGLShaderProgram::Attribute> position, textureCoordIn;
    
  private:
    static OpenGLShaderProgram::Attribute* createAttribute (OpenGLShaderProgram& shader,
                                                            const char* attributeName)
    {
      using namespace ::juce::gl;
      
      if (glGetAttribLocation (shader.getProgramID(), attributeName) < 0)
        return nullptr;
      
      return new OpenGLShaderProgram::Attribute (shader, attributeName);
    }
  };
  
  //==============================================================================
  // This class just manages the uniform values that the demo shaders use.
  struct Uniforms
  {
    explicit Uniforms (OpenGLShaderProgram& shaderProgram)
    {
      projectionMatrix.reset (createUniform (shaderProgram, "projectionMatrix"));
      texture         .reset (createUniform (shaderProgram, "demoTexture"));
    }
    
    std::unique_ptr<OpenGLShaderProgram::Uniform> projectionMatrix, texture;
    
  private:
    static OpenGLShaderProgram::Uniform* createUniform (OpenGLShaderProgram& shaderProgram,
                                                        const char* uniformName)
    {
      using namespace ::juce::gl;
      
      if (glGetUniformLocation (shaderProgram.getProgramID(), uniformName) < 0)
        return nullptr;
      
      return new OpenGLShaderProgram::Uniform (shaderProgram, uniformName);
    }
  };
  
  //==============================================================================
  /** This loads a 3D model from an OBJ file and converts it into some vertex buffers
   that we can draw.
   */
  struct SquareBuffer
  {
    explicit SquareBuffer ()
    {
      using namespace ::juce::gl;
      
      
      glGenBuffers (1, &vertexBuffer);
      glBindBuffer (GL_ARRAY_BUFFER, vertexBuffer);
      
      Array<Vertex> vertices;
      Array<juce::uint32> indices;
      
      indices.add(0);
      indices.add(2);
      indices.add(3);
      indices.add(0);
      indices.add(3);
      indices.add(1);
      
      numIndices = indices.size();
      
      Vertex top_left;
      top_left.position[0] = -1.0f;
      top_left.position[1] = -1.0f;
      top_left.position[2] = 0.0f;
      top_left.texCoord[0] = 0.0f;
      top_left.texCoord[1] = 0.0f;
      vertices.add(top_left);
      
      Vertex top_right;
      top_right.position[0] = 1.0f;
      top_right.position[1] = -1.0f;
      top_right.position[2] = 0.0f;
      top_right.texCoord[0] = 1.0f;
      top_right.texCoord[1] = 0.0f;
      vertices.add(top_right);
      
      Vertex bottom_left;
      bottom_left.position[0] = -1.0f;
      bottom_left.position[1] =  1.0f;
      bottom_left.position[2] = 0.0f;
      bottom_left.texCoord[0] = 0.0f;
      bottom_left.texCoord[1] = 1.0f;
      vertices.add(bottom_left);
      
      Vertex bottom_right;
      bottom_right.position[0] = 1.0f;
      bottom_right.position[1] = 1.0f;
      bottom_right.position[2] = 0.0f;
      bottom_right.texCoord[0] = 1.0f;
      bottom_right.texCoord[1] = 1.0f;
      vertices.add(bottom_right);
      
      
      glBufferData (GL_ARRAY_BUFFER,
                    static_cast<GLsizeiptr> (static_cast<size_t> (vertices.size()) * sizeof (Vertex)),
                    vertices.getRawDataPointer(), GL_STATIC_DRAW);
      
      glGenBuffers (1, &indexBuffer);
      glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
      glBufferData (GL_ELEMENT_ARRAY_BUFFER,
                    static_cast<GLsizeiptr> (static_cast<size_t> (numIndices) * sizeof (juce::uint32)),
                    indices.getRawDataPointer(), GL_STATIC_DRAW);
    }
    
    ~SquareBuffer()
    {
      using namespace ::juce::gl;
      
      glDeleteBuffers (1, &vertexBuffer);
      glDeleteBuffers (1, &indexBuffer);
    }
    
    void bind()
    {
      using namespace ::juce::gl;
      
      glBindBuffer (GL_ARRAY_BUFFER, vertexBuffer);
      glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    }
    
    GLuint vertexBuffer, indexBuffer;
    int numIndices;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SquareBuffer)
  };
  
  
  
  const char* vertexShader;
  const char* fragmentShader;
  
  
  std::unique_ptr<OpenGLShaderProgram> shader;
  std::unique_ptr<SquareBuffer> squareBuffer;
  std::unique_ptr<Attributes> attributes;
  std::unique_ptr<Uniforms> uniforms;
  
  juce::String newVertexShader, newFragmentShader;  
  
  Rectangle<int> bounds;
  float desktopScale;
  Rectangle<int> pixel_bounds;
  
  CriticalSection mutex;
  
  cv::Mat image;
  OpenGLTexture texture;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
