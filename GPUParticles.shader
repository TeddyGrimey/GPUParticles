Shader "Hidden/GPUParticles"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _GrainMultiplier ("Grain Multiplier", float) = 1
        _GrainScale ("Grain Scale", float) = 0
    }
    SubShader
    {
        Cull Off
        Tags {"Queue"="AlphaTest" "IgnoreProjector"="True" "RenderType"="Transparent"}
        
        Pass
        {
            Tags {"LightMode"="ForwardBase"}
            ZWrite On ZTest LEqual Cull Off
            Blend SrcAlpha OneMinusSrcAlpha
            
            Lighting On
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.0 
            #pragma multi_compile_fwdbase
            #pragma fragmentoption ARB_precision_hint_fastest

            #include "UnityCG.cginc"
            #include "ShaderTools.cginc"
            #include "AutoLight.cginc"
            #include "Lighting.cginc"
             
 
            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float4 color : COLOR;
                float4 texcoord1 : TEXCOORD1;
                float4 texcoord2 : TEXCOORD2;
                
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float3 normal : NORMAL;
                float3 worldPos : TEXCOORD1;
                float4 scrPos : TEXCOORD2;
                float3 centre : TEXCOORD3;
                float4 random : TEXCOORD4;
                LIGHTING_COORDS(5,6)
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            //TEXTURE2D_SAMPLER2D(_CameraDepthTexture, sampler_CameraDepthTexture); 
            sampler2D _CameraDepthTexture;

            sampler2D _SnowTexture;

            half4 _Color1;
            half4 _Color2;
            half _Smoothness;
            half _Metallic;
            half3 _Emission;

            uint _TriangleCount;
            float4 _LocalTime;
            float4 _LocalTimeNoise;
            int _Extent;
            float _NoiseAmplitude;
            float _NoiseFrequency;
            float3 _NoiseOffset;

            float3 _WindStrength;
            float3 _WindNoiseStrength;
            float _SizeMin, _SizeMax;
            float _Length;
            float _AlphaBlur;
            float3 _BoundsSize;
            float3 _BoundsOffset;
            float _StartZ;

            float4x4 _LocalToWorld;
            float4x4 _WorldToLocal;

            float4 _WorldSpaceLightPos1;

            struct ComputeParticleData
            {
                float3 position;
                float3 oldPosition;
                float3 velocity;
                float size;
            };

            StructuredBuffer<ComputeParticleData> _ComputeParticleData;

            uint Hash(uint s)
            {
                s ^= 2747636419u;
                s *= 2654435769u;
                s ^= s >> 16;
                s *= 2654435769u;
                s ^= s >> 16;
                s *= 2654435769u;
                return s;
            }

            float Random(float seed)
            {
                return float(Hash(seed)) / 4294967295.0; // 2^32-1
            }

            v2f vert (appdata v, uint vid : SV_VertexID)
            {
                v2f o;
                uint t_idx = vid / 3;         // Triangle index
                uint v_idx = vid - t_idx * 3; // Vertex index


                uint seed = (float)t_idx / _TriangleCount;
                seed = ((seed << 16) + t_idx) * 4;

                float row = t_idx / _Extent;
                float col = (t_idx % _Extent) / 2;

                float boxSize = 16;

                float u = _ComputeParticleData[t_idx].position.x;
                float z = _ComputeParticleData[t_idx].position.y;
                float t = _ComputeParticleData[t_idx].position.z;



                float randX = Random(t_idx);
                float randY = Random(t_idx + 1);
                float randZ = Random(t_idx + 2);

                float size = lerp(_SizeMin * 1.8, _SizeMax * 1.8, randX);

                size *=  InverseLerp(distance(_WorldSpaceCameraPos,float3(u, z,t)), 0,3);


                o.random = 1;
                o.random.x = randX;
                o.random.y = randY;
                

                float3 viewDir = normalize(_WorldSpaceCameraPos - float3(u, z,t));
                float3 windDir = minMagnitude(_ComputeParticleData[t_idx].velocity, 0.5);
                float3 crossWindView = normalize(cross(windDir, viewDir));
                float VdotW = smoothstep(0.0,0.0001, 1 - abs(dot(viewDir, windDir)));

                //windDir.x *= randX;

                float3 v1 = -crossWindView * size;
                float3 v2 = windDir * -_Length * size;// * 1.5;
                float3 v3 = crossWindView  * size; 

                //v2 = lerp(normalize( -cross(viewDir, crossWindView)) * size * 1.5, v2, VdotW);

                float3 windNormalized = normalize(_WindStrength);
                float viewDotWind = acos( dot(normalize(_WorldSpaceCameraPos.xyz - o.centre.xyz), windNormalized.xyz));
                float angle = atan2(normalize(_WorldSpaceCameraPos.xyz), windNormalized.xyz);

                v1.xyz += float3(u, z,t); 
                v2.xyz += float3(u, z,t);
                v3.xyz += float3(u, z,t);

                o.centre = lerp( ((v1 + v3) / 2)- mul(unity_WorldToObject, float4(0,0,0,1)),  (v2) - mul(unity_WorldToObject, float4(0,0,0,1)),  0);
                v.vertex.xyz = lerp(lerp(v3,v2,is_equal(v_idx, 1)),v1,is_equal(v_idx, 0));

                o.color = 0;
                o.color.r = (is_equal(v_idx, 0));
                o.color.g = (is_equal(v_idx, 1));
                o.color.b = (is_equal(v_idx, 2));
                o.color.a = 1;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.scrPos = ComputeScreenPos(o.pos);
                o.normal = saturate(smoothstep(0,1, normalize(v.vertex.xyz - o.centre)));
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                TRANSFER_VERTEX_TO_FRAGMENT(o);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = 1;
                float dist = distance(i.centre, i.worldPos);

                float angle = atan2(normalize(i.worldPos.xy - i.centre.xy).x, normalize(i.worldPos.xy - i.centre.xy).y);//acos(dot(normalize(i.worldPos.xy - i.centre.xy), float2(0,1)) ) ; 
 
                //float size = lerp(_SizeMin, _SizeMax, i.random.x) * lerp(lerp(0.4,1, InverseLerp(snoise(float2(angle * 0.5, i.random.y * 100)), -1,1)), 1, 0);//i.color.g);
                float size = lerp(_SizeMin, _SizeMax, i.random.x);
                size *= smoothstep(0,1,InverseLerp(distance(_WorldSpaceCameraPos, i.worldPos), 0.5,3));
                size += InverseLerp(distance(_WorldSpaceCameraPos,i.worldPos), 0, 30) * 0.04 * _SizeMax;
                
                float fadeToCamera = InverseLerp(distance(_WorldSpaceCameraPos,i.worldPos), 0, 5);
                float particleAlpha = less_than(lerp(0.2, 0.16, fadeToCamera), smoothstep(0.0,0.06,i.color.r * i.color.g * i.color.b)) * smoothstep(0,_AlphaBlur, (1 - i.color.g));
                float depthSample  = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, i.scrPos)).r;
                float newDepth = depthSample - i.scrPos.w;
                col.a = particleAlpha;
                col.a *= step(distance(i.worldPos, _WorldSpaceCameraPos.xyz), depthSample.r);
                col.a *= smoothstep(0.2,0.8,clamp(newDepth * 1, 0,1));
                if(col.a < 0.01)
                    discard;

                float3 lightDir = normalize(_WorldSpaceLightPos0.xyz);
                float shadowAttenuation = LIGHT_ATTENUATION(i);
                //UNITY_LIGHT_ATTENUATION(attenuation, i, i.worldPos.xyz);

                col.rgba *= lerp(_Color1, _Color2, i.random.x);
                col.rgb *= shadowAttenuation * _LightColor0.rgb;
                return col;
            }
            ENDCG
        }
    }
}
